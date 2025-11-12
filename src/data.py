# src/data.py
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import ray
from numpy import ndarray
from ray.data import Dataset

logger = logging.getLogger(__name__)

# Constants
FMNIST_MEAN = 0.2860
FMNIST_STD = 0.3530


def normalize_images(images: ndarray) -> ndarray:
    """Normalize image(s) for training and inference.

    Args:
        images: Shape (28, 28) or (B, 28, 28) uint8 [0, 255]

    Returns:
        Normalized images shape (1, 28, 28) or (B, 1, 28, 28) float32
    """
    x = images.astype(np.float32) / 255.0
    x = (x - FMNIST_MEAN) / FMNIST_STD

    # Add channel dimension
    if x.ndim == 2:  # Single image (28, 28)
        return x[None, :, :]  # (1, 28, 28)
    else:  # Batch (B, 28, 28)
        return x[:, None, :, :]  # (B, 1, 28, 28)


def load_data(
    data_path: Path | str,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Load and transform datasets from Parquet files."""
    logger.info(f"Loading data from {data_path}")

    data_path = Path(data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Parquet data not found at {data_path}")

    train_ds = ray.data.read_parquet(str(train_path))
    val_ds = ray.data.read_parquet(str(val_path))

    if limit_train:
        train_ds = train_ds.limit(limit_train)
    if limit_val:
        val_ds = val_ds.limit(limit_val)

    # Apply vectorized normalization
    train_ds = train_ds.map_batches(
        lambda batch: {
            "image": normalize_images(batch["image"]),
            "label": batch["label"],
        },
        batch_format="numpy",
    )
    val_ds = val_ds.map_batches(
        lambda batch: {
            "image": normalize_images(batch["image"]),
            "label": batch["label"],
        },
        batch_format="numpy",
    )

    return train_ds, val_ds


if __name__ == "__main__":
    data_dir = Path("/workspace/project/data/FashionMNIST_parquet")
    train_ds, val_ds = load_data(data_dir, limit_train=100, limit_val=50)
    print(f"Train: {train_ds.count()} samples")
    print(f"Val: {val_ds.count()} samples")
    print("Schema:", train_ds.schema())

    # Test single image normalization
    sample = train_ds.take(1)[0]
    print(f"Sample image shape: {sample['image'].shape}")
