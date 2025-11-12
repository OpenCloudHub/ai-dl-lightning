from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    # For experiment tracking
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "fashion-mnist"
    mlflow_registered_model_name: str = "dev.fashion-mnist-classifier"
    # Storage paths
    ray_storage_path: str = "/tmp/ray_results"
    # For workflow tagging
    argo_workflow_uid: str | None = "TEST"


# Singleton config instance
BASE_CNFG = BaseConfig()
