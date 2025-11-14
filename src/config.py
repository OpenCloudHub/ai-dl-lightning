from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
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
    dvc_metrics_path: str = "data/fashion-mnist/processed/metadata.json"

    # Optional: Workflow metadata (only set during training in Argo)
    argo_workflow_uid: str | None = None
    docker_image_tag: str | None = None


# Singleton config instance
BASE_CNFG = BaseConfig()
