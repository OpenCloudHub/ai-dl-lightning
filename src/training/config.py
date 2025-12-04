# ==============================================================================
# Training Configuration
# ==============================================================================
#
# Centralized configuration for the training pipeline using pydantic-settings.
#
# All settings can be overridden via environment variables (uppercase with
# underscores, e.g., MLFLOW_TRACKING_URI, RAY_NUM_WORKERS).
#
# Configuration Categories:
#   - MLflow: Tracking URI, experiment name, model registry name
#   - Ray: Storage endpoint, checkpoint path, worker count
#   - DVC: Repository URL, data paths for train/val/metadata
#
# Environment Files:
#   - .env.docker: Local Docker Compose development
#   - .env.minikube: Minikube/Kubernetes development
#
# Usage:
#   from src.training.config import TRAINING_CONFIG
#   print(TRAINING_CONFIG.mlflow_experiment_name)
#
# ==============================================================================

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Training configuration loaded from environment variables."""

    # For experiment tracking
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "fashion-mnist"
    mlflow_registered_model_name: str = "dev.fashion-mnist-classifier"

    # Ray options
    ray_storage_endpoint: str = "http://minio.minio-tenant.svc.cluster.local:80"
    ray_storage_path: str = "ray-results"
    ray_num_workers: int = 1

    # DVC repository URL
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_train_data_path: str = "data/fashion-mnist/processed/train/train.parquet"
    dvc_val_data_path: str = "data/fashion-mnist/processed/val/val.parquet"
    dvc_metrics_path: str = "data/fashion-mnist/metadata.json"


# Singleton instances
TRAINING_CONFIG = TrainingConfig()
