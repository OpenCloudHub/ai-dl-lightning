from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Training configuration loaded from environment variables."""

    # For experiment tracking
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "fashion-mnist"
    mlflow_registered_model_name: str = "dev.fashion-mnist-classifier"

    # Storage paths
    ray_storage_path: str = "/tmp/ray_results"

    # DVC repository URL
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_train_data_path: str = "data/fashion-mnist/processed/train/train.parquet"
    dvc_val_data_path: str = "data/fashion-mnist/processed/val/val.parquet"
    dvc_metrics_path: str = "data/fashion-mnist/metadata.json"


# Singleton instances
TRAINING_CONFIG = TrainingConfig()
