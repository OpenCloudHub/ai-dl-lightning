# ==============================================================================
# Serving API Schemas
# ==============================================================================
#
# Pydantic models for Fashion MNIST API request/response validation.
#
# Schema Overview:
#   - PredictionRequest: Input validation for batch image predictions
#   - PredictionResponse: Structured output with predictions and metadata
#   - ModelInfo: Model metadata including normalization parameters
#   - HealthResponse: Health check status information
#
# Input Formats Supported:
#   - Flattened: (batch_size, 784) - 28x28 images flattened to 784 integers
#   - 2D: (batch_size, 28, 28) - 28x28 grayscale images
#   - Pixel values: 0-255 (uint8)
#
# Fashion MNIST Classes (0-9):
#   T-shirt/top, Trouser, Pullover, Dress, Coat,
#   Sandal, Shirt, Sneaker, Bag, Ankle boot
#
# Validation:
#   - Batch size limited to REQUEST_MAX_LENGTH (default: 1000)
#   - Pixel values validated to be in [0, 255] range
#   - Shape validation for both flattened and 2D formats
#
# ==============================================================================

"""Schema definitions for the Fashion MNIST serving module."""

from datetime import datetime
from enum import StrEnum, auto
from typing import Annotated, List

import numpy as np
from pydantic import AfterValidator, BaseModel, Field

from src._utils.logging import get_logger
from src.serving.config import SERVING_CONFIG

logger = get_logger(__name__)

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def validate_images(
    images: List[List[int]] | List[List[List[int]]],
) -> List[List[int]] | List[List[List[int]]]:
    """Validate input images."""
    arr = np.asarray(images, dtype=np.uint8)

    # Check shape - accept either flattened (batch, 784) or 2D (batch, 28, 28)
    if arr.ndim == 2 and arr.shape[1] == 784:
        # Flattened format is OK
        pass
    elif arr.ndim == 3 and arr.shape[1:] == (28, 28):
        # 2D format is OK
        pass
    else:
        raise ValueError(
            f"Invalid image shape {arr.shape}. Expected (batch, 784) or (batch, 28, 28)"
        )

    # Check value range [0, 255]
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(
            f"Pixel values must be in range [0, 255], got [{arr.min()}, {arr.max()}]"
        )

    return images


class PredictionRequest(BaseModel):
    """Input model for predictions with validation."""

    images: Annotated[
        List[List[int]] | List[List[List[int]]],
        AfterValidator(validate_images),
        Field(
            min_length=1,
            max_length=SERVING_CONFIG.request_max_length,  # Prevent DOS attacks with huge batches
            description="List of grayscale images. Each image can be either:\n"
            "- Flattened: 784 integers (28x28 flattened)\n"
            "- 2D: 28x28 array of integers\n"
            "Pixel values must be in range [0, 255] (uint8).",
            examples=[
                # Example with flattened images
                [[0] * 784, [128] * 784],
                # Example with 2D images
                [[[0] * 28 for _ in range(28)], [[128] * 28 for _ in range(28)]],
            ],
        ),
    ]


class Prediction(BaseModel):
    """Single prediction result."""

    class_id: int = Field(..., description="Predicted class ID (0-9)", ge=0, le=9)
    class_name: str = Field(
        ..., description="Human-readable class name (e.g., 'T-shirt/top', 'Trouser')"
    )
    confidence: float = Field(
        ..., description="Prediction confidence (0-1)", ge=0.0, le=1.0
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input image"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")
    processing_time_ms: float = Field(
        ..., description="Time taken to process request in milliseconds"
    )


class ModelInfo(BaseModel):
    """Model metadata information."""

    model_uri: str = Field(..., description="URI of the model used")
    model_uuid: str = Field(..., description="MLflow model UUID")
    run_id: str = Field(..., description="MLflow run ID associated with the model")
    model_signature: dict | None = Field(None, description="MLflow model signature")
    data_version: str | None = Field(
        None, description="DVC data version used for training"
    )
    training_timestamp: datetime | None = Field(
        None, description="When the model was trained"
    )
    expected_input_shape: str = Field(
        default="(batch_size, 28, 28) or (batch_size, 784)",
        description="Expected input shape for images",
    )
    output_classes: List[str] = Field(
        default_factory=lambda: list(FASHION_MNIST_CLASSES),
        description="List of output class names",
    )
    normalization_params: dict | None = Field(
        None, description="Normalization parameters (mean, std) used by the model"
    )


class APIStatus(StrEnum):
    """API status enumeration."""

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: APIStatus = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_uri: str | None = Field(None, description="Current model URI")
    uptime_seconds: int | None = Field(None, description="Service uptime in seconds")


class RootResponse(BaseModel):
    """Response model for root endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    docs: str = Field(..., description="URL to API documentation")
    health: str = Field(..., description="URL to health check endpoint")


class ErrorDetail(BaseModel):
    """Error detail model."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str | ErrorDetail = Field(..., description="Error details")
