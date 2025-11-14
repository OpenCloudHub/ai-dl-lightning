# src/data.py
import json
from typing import Optional, Tuple

import dvc.api
import numpy as np
import ray
from numpy import ndarray
from ray.data import Dataset

from src._utils.logging import get_logger, log_section
from src.config import BASE_CNFG

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
        BASE_CNFG.dvc_metrics_path,
        repo=BASE_CNFG.dvc_repo,
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
        f"Loading data version '{version}' from DVC repo '{BASE_CNFG.dvc_repo}'"
    )
    log_section(f"Loading Data Version {version}", "📦")

    logger.info(f"DVC repo: [cyan]{BASE_CNFG.dvc_repo}[/cyan]")

    # Get URLs from DVC
    train_path = dvc.api.get_url(
        BASE_CNFG.dvc_train_data_path,
        repo=BASE_CNFG.dvc_repo,
        rev=version,
    )
    val_path = dvc.api.get_url(
        BASE_CNFG.dvc_val_data_path,
        repo=BASE_CNFG.dvc_repo,
        rev=version,
    )
    metadata_content = dvc.api.read(
        BASE_CNFG.dvc_metrics_path, repo=BASE_CNFG.dvc_repo, rev=version
    )
    metadata = json.loads(metadata_content)
    logger.info(
        f"Loaded dataset: [bold]{metadata['dataset']['name']}[/bold] [green]({version})[/green]"
    )

    # Extract normalization statistics from metadata
    mean = metadata["metrics"]["train"]["pixel_mean"]
    std = metadata["metrics"]["train"]["pixel_std"]
    logger.info(
        f"Normalization: mean=[yellow]{mean:.4f}[/yellow], std=[yellow]{std:.4f}[/yellow]"
    )

    # Load datasets
    train_ds = ray.data.read_parquet(train_path)
    val_ds = ray.data.read_parquet(val_path)

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

    logger.success(
        f"✨ Data loaded: {train_ds.count()} train, {val_ds.count()} val samples"
    )

    return train_ds, val_ds, metadata
