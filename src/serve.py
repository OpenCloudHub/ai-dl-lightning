# src/serve.py
from datetime import datetime, timezone
from typing import List

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from torchvision.datasets import FashionMNIST

from src._utils.logging import get_logger
from src.data import get_normalization_params, normalize_images

logger = get_logger(__name__)

FASHION_MNIST_CLASSES = FashionMNIST.classes

app = FastAPI(
    title="👕 Fashion MNIST Classifier Demo API",
    description="Fashion MNIST classification using Ray Serve + MLflow",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    images: List[List[int]] | List[List[List[int]]] = Field(
        ...,
        description="Batch of images. Each: either 784 ints (flattened) or 28x28 array [0-255].",
    )


class Prediction(BaseModel):
    """Single prediction result"""

    class_id: int = Field(..., description="Predicted class ID (0-9)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input image"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")


class ModelInfo(BaseModel):
    model_uri: str = Field(..., description="URI of the model used")
    model_uuid: str = Field(..., description="MLflow model UUID")
    run_id: str = Field(..., description="MLflow run ID associated with the model")
    data_version: str = Field(..., description="DVC data version used for training")
    model_signature: str | None = Field(None, description="MLflow model signature")
    expected_input_shape: str = Field(
        "(batch_size, 28, 28) or (batch_size, 784)",
        description="Expected input tensor shape",
    )
    output_classes: List[str] = Field(
        default=list(FASHION_MNIST_CLASSES),
        description="List of possible output classes",
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(..., description="API health status")
    model_info: ModelInfo | None = Field(
        None, description="Information about the loaded model"
    )


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 3},
)
@serve.ingress(app)
class FashionMNISTClassifier:
    def __init__(self, model_uri: str | None = None):
        """Initialize the classifier, optionally with a model URI."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = FASHION_MNIST_CLASSES
        self.model = None
        self.model_info: ModelInfo | None = None
        self.data_version: str | None = None
        self.norm_mean: float | None = None
        self.norm_std: float | None = None

        # Load model if URI provided at init
        if model_uri:
            self._load_model(model_uri)

    def _load_model(self, model_uri: str):
        """Internal method to load model and fetch normalization parameters."""
        logger.info(f"📦 Loading model from: {model_uri}")

        # Get model info from MLflow
        info = mlflow.models.get_model_info(model_uri)

        # Fetch the data version from the model's training run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(info.run_id)

        # Get data version from run parameters
        self.data_version = run.data.tags.get("dvc_data_version")
        if not self.data_version:
            raise ValueError(
                f"Model at {model_uri} was trained without dvc_data_version parameter. "
                "Cannot determine which normalization to use."
            )

        logger.info(f"📊 Using normalization from DVC version: {self.data_version}")

        # Fetch normalization parameters from DVC
        self.norm_mean, self.norm_std = get_normalization_params(self.data_version)
        logger.info(
            f"   Normalization: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}"
        )

        # Load the model
        self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
        self.model.eval()

        # Build ModelInfo
        self.model_info = ModelInfo(
            model_uri=model_uri,
            model_uuid=info.model_uuid,
            run_id=info.run_id,
            data_version=self.data_version,
            model_signature=str(info.signature) if info.signature else None,
        )

        logger.info(f"✅ Model loaded successfully on device: {self.device}")
        logger.info(f"   Model UUID: {self.model_info.model_uuid}")
        logger.info(f"   Run ID: {self.model_info.run_id}")
        logger.info(f"   Data version: {self.data_version}")

    def reconfigure(self, config: dict) -> None:
        """Handle model updates without restarting the deployment.

        Also handles initial load if model wasn't provided at __init__.
        Update via: serve.run(..., user_config={"model_uri": "new_uri"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet (first time), load it
        if self.model is None or self.model_info is None:
            logger.info("🆕 Initial model load")
            self._load_model(new_model_uri)
            return

        # If model already loaded, check if URI changed
        if new_model_uri == self.model_info.model_uri:
            logger.info("ℹ️ No model update needed (same URI)")
            return

        # URI changed, reload model
        logger.info(
            f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
        )
        self._load_model(new_model_uri)

    @app.get("/", response_model=HealthResponse, summary="Health Check")
    async def root(self):
        """Health check endpoint."""
        if self.model_info is None:
            return HealthResponse(
                status="not_ready",
                model_info=None,
            )

        return HealthResponse(
            status="healthy",
            model_info=self.model_info,
        )

    @app.post(
        "/predict", response_model=PredictionResponse, summary="Predict Fashion Item"
    )
    async def predict(self, request: PredictionRequest):
        """
        Predict fashion item class from images.

        Images should be 28x28 grayscale, values 0-255 (uint8).
        Accepts both flattened (784,) and 2D (28, 28) formats.
        Normalization is applied automatically using the same transform as training.
        """
        # Check if model is loaded
        if self.model is None or self.model_info is None:
            raise HTTPException(
                503, "Model not loaded. Configure the deployment with a model_uri."
            )

        try:
            # Convert to numpy array
            arr = np.asarray(request.images, dtype=np.uint8)

            # Handle different input shapes
            if arr.ndim == 2 and arr.shape[1] == 784:
                # Flattened: (batch_size, 784) -> (batch_size, 28, 28)
                arr = arr.reshape(-1, 28, 28)
            elif arr.ndim == 3 and arr.shape[1:] == (28, 28):
                # Already correct: (batch_size, 28, 28)
                pass
            else:
                raise HTTPException(
                    400,
                    f"Expected shape (batch_size, 784) or (batch_size, 28, 28), got {arr.shape}",
                )

            # Apply the SAME normalization as training (reusing normalize_images!)
            normalized = normalize_images(
                arr, mean=self.norm_mean, std=self.norm_std
            )  # -> (B, 1, 28, 28) float32

            # Convert to torch and move to device
            batch = torch.from_numpy(normalized).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                confidences, class_ids = torch.max(probs, dim=1)

            # Build predictions
            predictions = [
                Prediction(
                    class_id=int(cls_id),
                    class_name=self.class_names[int(cls_id)],
                    confidence=float(conf),
                )
                for cls_id, conf in zip(
                    class_ids.cpu().numpy(), confidences.cpu().numpy()
                )
            ]

            return PredictionResponse(
                predictions=predictions,
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise HTTPException(500, f"Prediction failed: {e}")


class AppBuilderArgs(BaseModel):
    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/my_model/1)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI."""
    return FashionMNISTClassifier.bind(model_uri=args.model_uri)
