"""Fashion MNIST serving application using Ray Serve + MLflow."""

from datetime import datetime, timezone

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from src._utils.logging import get_logger
from src.serving.schemas import (
    FASHION_MNIST_CLASSES,
    APIStatus,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)
from src.training.data import get_normalization_params, normalize_images

logger = get_logger(__name__)

app = FastAPI(
    title="👕 Fashion MNIST Classifier API",
    description="Fashion MNIST classification using Ray Serve + MLflow + PyTorch Lightning",
    version="1.0.0",
)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(app)
class FashionMNISTClassifier:
    def __init__(self, model_uri: str | None = None) -> None:
        """Initialize the classifier, optionally with a model URI."""
        logger.info("👕 Initializing Fashion MNIST Classifier Service")
        self.status = APIStatus.NOT_READY
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = FASHION_MNIST_CLASSES
        self.model = None
        self.model_info: ModelInfo | None = None
        self.data_version: str | None = None
        self.norm_mean: float | None = None
        self.norm_std: float | None = None
        self.start_time = datetime.now(timezone.utc)

        # Load model if URI provided at init
        if model_uri:
            try:
                self._load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model during initialization: {e}")
                self.status = APIStatus.UNHEALTHY

    def _load_model(self, model_uri: str) -> None:
        """Internal method to load model and fetch metadata."""
        logger.info(f"📦 Loading model from: {model_uri}")
        self.status = APIStatus.LOADING

        try:
            # Get model info first to validate URI
            info = mlflow.models.get_model_info(model_uri)

            # Get training run metadata
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(info.run_id)

            # Get data version from run tags
            self.data_version = run.data.tags.get("dvc_data_version")
            if not self.data_version:
                logger.warning("No dvc_data_version found in run tags")
                raise ValueError(
                    f"Model at {model_uri} was trained without dvc_data_version tag. "
                    "Cannot determine normalization parameters."
                )

            logger.info(f"📊 Using normalization from DVC version: {self.data_version}")

            # Fetch normalization parameters from DVC
            self.norm_mean, self.norm_std = get_normalization_params(self.data_version)
            logger.info(
                f"   Normalization: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}"
            )

            # Load the PyTorch model
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()

            # Extract training timestamp
            training_timestamp = datetime.fromtimestamp(
                run.info.start_time / 1000.0, tz=timezone.utc
            )

            # Build ModelInfo
            self.model_info = ModelInfo(
                model_uri=model_uri,
                model_uuid=info.model_uuid,
                run_id=info.run_id,
                model_signature=info.signature.to_dict() if info.signature else None,
                data_version=self.data_version,
                training_timestamp=training_timestamp,
                normalization_params={
                    "mean": self.norm_mean,
                    "std": self.norm_std,
                },
            )

            self.status = APIStatus.HEALTHY
            logger.success("✅ Model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Model UUID: {self.model_info.model_uuid}")
            logger.info(f"   Run ID: {self.model_info.run_id}")
            logger.info(f"   Data version: {self.data_version}")

        except mlflow.exceptions.MlflowException as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ MLflow error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model from MLflow: {str(e)}",
            )
        except Exception as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ Unexpected error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error loading model: {str(e)}",
            )

    def reconfigure(self, config: dict) -> None:
        """Handle model updates without restarting the deployment.

        Check: https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html

        Update via: serve.run(..., user_config={"model_uri": "new_uri"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet, load it
        if self.model_info is None:
            logger.info("🆕 Initial model load via reconfigure")
            self._load_model(new_model_uri)
            return

        # Check if URI changed
        if self.model_info.model_uri != new_model_uri:
            logger.info(
                f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
            )
            self._load_model(new_model_uri)
        else:
            logger.info("ℹ️ Model URI unchanged, skipping reload")

    @app.get(
        "/",
        response_model=RootResponse,
        summary="Root endpoint",
        responses={
            200: {"description": "Service information"},
            503: {"description": "Service not healthy"},
        },
    )
    async def root(self):
        """Root endpoint with basic info."""
        return RootResponse(
            service="Fashion MNIST Classifier API",
            version="1.0.0",
            status=self.status.value,
            docs="/docs",
            health="/health",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        responses={
            200: {"description": "Service is healthy"},
            503: {"description": "Service is not ready or unhealthy"},
        },
    )
    async def health(self):
        """Health check endpoint."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        response = HealthResponse(
            status=self.status,
            model_loaded=self.model is not None,
            model_uri=self.model_info.model_uri if self.model_info else None,
            uptime_seconds=int(uptime),
        )

        # Return 503 if not healthy
        if self.status != APIStatus.HEALTHY:
            detail = response.model_dump()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            )

    @app.get(
        "/info",
        response_model=ModelInfo,
        summary="Model Information",
        responses={
            200: {"description": "Model information"},
            503: {"description": "Model not loaded", "model": ErrorResponse},
        },
    )
    async def info(self):
        """Get detailed model information including normalization parameters."""
        if self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please configure the deployment with a model_uri.",
            )
        return self.model_info

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict Fashion Item",
        responses={
            200: {"description": "Successful prediction"},
            400: {"description": "Invalid input", "model": ErrorResponse},
            503: {"description": "Model not loaded", "model": ErrorResponse},
            500: {"description": "Internal server error", "model": ErrorResponse},
        },
    )
    async def predict(self, request: PredictionRequest):
        """
        Predict fashion item class from grayscale images.

        **Input Format:**
        - Images should be 28x28 grayscale, values 0-255 (uint8)
        - Accepts both flattened (784,) and 2D (28, 28) formats
        - Batch size: 1-100 images

        **Output:**
        - class_id: 0-9 representing the predicted class
        - class_name: One of ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        - confidence: Softmax probability of the predicted class (0-1)

        **Note:** Normalization is applied automatically using the same parameters as training.
        """
        # Check if model is loaded
        if self.model is None or self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Configure the deployment with a model_uri.",
            )

        start_time = datetime.now(timezone.utc)

        try:
            # Convert to numpy array (validation already done by Pydantic)
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
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Expected shape (batch_size, 784) or (batch_size, 28, 28), got {arr.shape}",
                )

            # Apply the SAME normalization as training
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

            # Calculate processing time
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return PredictionResponse(
                predictions=predictions,
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_time,
            )

        except HTTPException:
            raise
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}",
            )
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


class AppBuilderArgs(BaseModel):
    """Arguments for building the Ray Serve application."""

    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/ci.fashion-mnist-classifier/1 or runs:/run_id/model)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI.

    Examples:
        Basic usage:
        >>> serve run src.serving.serve:app_builder model_uri="models:/ci.fashion-mnist-classifier/15"

        With hot reload for development:
        >>> serve run src.serving.serve:app_builder model_uri="models:/ci.fashion-mnist-classifier/15" --reload

    Args:
        args: Configuration arguments including model URI

    Returns:
        Ray Serve Application ready to deploy
    """
    return FashionMNISTClassifier.bind(model_uri=args.model_uri)
