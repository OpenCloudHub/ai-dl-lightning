# ==============================================================================
# DVC Data Loading Module
# ==============================================================================
#
# Loads versioned Fashion MNIST data from DVC remote storage (S3/MinIO).
#
# Key Features:
#   - Fetches data from external DVC repository (data-registry)
#   - Retrieves normalization parameters (mean, std) from DVC metadata
#   - Returns Ray Datasets for distributed training with sharding
#
# Data Flow:
#   1. Resolve DVC paths to S3 URLs using dvc.api
#   2. Load Parquet files via Ray Data with s3fs filesystem
#   3. Apply normalization using training set statistics
#   4. Return shardable Ray Datasets for distributed training
#
# Normalization:
#   - Statistics (mean, std) are stored in DVC metadata.json
#   - Same parameters are used in serving for consistency
#   - normalize_images() function is shared between training and serving
#
# Usage:
#   train_ds, val_ds, metadata = load_data(
#       version="fashion-mnist-v1.0.0",
#       limit_train=1000,  # Optional: limit for quick testing
#   )
#
# Environment Variables:
#   - AWS_ACCESS_KEY_ID: S3/MinIO access key
#   - AWS_SECRET_ACCESS_KEY: S3/MinIO secret key
#   - AWS_ENDPOINT_URL: S3/MinIO endpoint URL
#
# See Also:
#   - https://github.com/OpenCloudHub/data-registry (DVC repo)
#   - src/serving/serve.py (uses get_normalization_params)
#
# ==============================================================================

import json
import os
from typing import Optional, Tuple

import dvc.api
import numpy as np
import ray
import s3fs
from numpy import ndarray
from ray.data import Dataset

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG

logger = get_logger(__name__)


def normalize_images(images: ndarray, mean: float, std: float) -> ndarray:
    """Normalize image(s) using dataset statistics.

    This function is used by both training and serving pipelines.

    Args:
        images: Shape (28, 28) or (B, 28, 28) uint8 [0, 255]
        mean: Pixel mean from training data statistics
        std: Pixel std from training data statistics

    Returns:
        Normalized images shape (1, 28, 28) or (B, 1, 28, 28) float32
    """
    # Convert to float and normalize to [0, 1]
    x = images.astype(np.float32) / 255.0

    # Apply dataset statistics
    x = (x - mean) / std

    # Add channel dimension
    if x.ndim == 2:  # Single image (28, 28)
        return x[None, :, :]  # (1, 28, 28)
    else:  # Batch (B, 28, 28)
        return x[:, None, :, :]  # (B, 1, 28, 28)


def get_normalization_params(version: str) -> Tuple[float, float]:
    """Fetch normalization parameters from DVC metadata.

    Args:
        version: DVC version tag (e.g., 'v0.0.3')

    Returns:
        Tuple of (mean, std) for normalization
    """
    metadata_content = dvc.api.read(
        TRAINING_CONFIG.dvc_metrics_path,
        repo=TRAINING_CONFIG.dvc_repo_url,
        rev=version,
    )
    metadata = json.loads(metadata_content)
    mean = metadata["metrics"]["train"]["pixel_mean"]
    std = metadata["metrics"]["train"]["pixel_std"]
    return mean, std


def load_data(
    version: str,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
) -> Tuple[Dataset, Dataset, dict]:
    """Load and transform datasets from DVC.

    Args:
        version: DVC version tag (e.g., 'v0.0.3')
        limit_train: Optional limit on training samples
        limit_val: Optional limit on validation samples

    Returns:
        Tuple of (train_ds, val_ds, metadata)
    """
    logger.info(
        f"Loading data version '{version}' from DVC repo '{TRAINING_CONFIG.dvc_repo_url}'"
    )
    log_section(f"Loading Data Version {version}", "📦")
    logger.info(f"DVC repo: {TRAINING_CONFIG.dvc_repo_url}")
    logger.info(f"DVC remote: {TRAINING_CONFIG.dvc_remote}")

    # Get URLs from DVC
    train_path = dvc.api.get_url(
        TRAINING_CONFIG.dvc_train_data_path,
        repo=TRAINING_CONFIG.dvc_repo_url,
        rev=version,
        remote=TRAINING_CONFIG.dvc_remote,
    )
    val_path = dvc.api.get_url(
        TRAINING_CONFIG.dvc_val_data_path,
        repo=TRAINING_CONFIG.dvc_repo_url,
        rev=version,
        remote=TRAINING_CONFIG.dvc_remote,
    )

    metadata_content = dvc.api.read(
        TRAINING_CONFIG.dvc_metrics_path,
        repo=TRAINING_CONFIG.dvc_repo_url,
        rev=version,
        remote=TRAINING_CONFIG.dvc_remote,
    )
    metadata = json.loads(metadata_content)

    logger.info(f"Loaded dataset: {metadata['dataset']['name']} ({version})")

    # Extract normalization statistics from metadata
    mean = metadata["metrics"]["train"]["pixel_mean"]
    std = metadata["metrics"]["train"]["pixel_std"]

    logger.info(f"Normalization: mean={mean:.4f}, std={std:.4f}")

    # Configure S3 filesystem using s3fs with SSL verification disabled
    s3_client = s3fs.S3FileSystem(
        anon=False,
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        client_kwargs={
            "verify": False,  # Disable SSL verification for self-signed certs
        },
    )

    # Load datasets with custom filesystem
    train_ds = ray.data.read_parquet(
        train_path, filesystem=s3_client, file_extensions=None
    )
    val_ds = ray.data.read_parquet(val_path, filesystem=s3_client, file_extensions=None)

    # Apply optional limits for quick testing or debugging
    if limit_train:
        train_ds = train_ds.limit(limit_train)
    if limit_val:
        val_ds = val_ds.limit(limit_val)

    # Apply normalization per row
    def normalize_row(row):
        """Normalize image and add channel dimension."""
        img = np.array(row["image"], dtype=np.float32)
        normalized = normalize_images(img, mean=mean, std=std)
        return {"image": normalized, "label": row["label"]}

    train_ds = train_ds.map(normalize_row)
    val_ds = val_ds.map(normalize_row)

    logger.success("✨ Data loaded")

    return train_ds, val_ds, metadata
