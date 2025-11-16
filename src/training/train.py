import argparse
from pathlib import Path

import lightning.pytorch as pl
import mlflow
import ray
import torch
from pydantic_settings import BaseSettings
from ray.train import (
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
    get_checkpoint,
    get_context,
    get_dataset_shard,
)
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG
from src.training.data import load_data
from src.training.model import SimpleImageClassifier

logger = get_logger(__name__)


# ============================================== #
# 🔹 SECTION: Data Contract
# ============================================== #
class WorkflowTags(BaseSettings):
    """⚠️ Data Contract for CI/CD Workflows:
    ============================================================
    When running in Argo Workflows, the following environment variables MUST be set
    and will take precedence over CLI arguments:

    - ARGO_WORKFLOW_UID: Unique identifier for the Argo workflow run
    - DOCKER_IMAGE_TAG: Docker image tag used for training (for reproducibility)
    - DVC_DATA_VERSION: Data version from DVC (takes precedence over --data-version arg)

    For local development, set these to "DEV" in your .env file.
    """

    # CI/CD Data Contract - Required fields set by Argo Workflows
    # These values are used for MLflow tagging and reproducibility
    argo_workflow_uid: str  # Must be set (use "DEV" for local development)
    docker_image_tag: str  # Must be set (use "DEV" for local development)
    dvc_data_version: str  # Takes precedence over --data-version CLI arg


WORKFLOW_TAGS = WorkflowTags()


# ============================================== #
# 🔹 SECTION: Training Functions
# ============================================== #
def train_fn_per_worker(train_loop_cnfg: dict):
    """Training code that runs on each worker."""
    worker_logger = get_logger(__name__)

    # Load data shard on each worker
    train_ds_shard = get_dataset_shard("train")
    val_ds_shard = get_dataset_shard("val")

    # Build data iterators
    train_iter = train_ds_shard.iter_torch_batches(
        batch_size=train_loop_cnfg.get("batch_size"),
        prefetch_batches=2,
        dtypes={"image": torch.float32, "label": torch.int64},
    )
    val_iter = val_ds_shard.iter_torch_batches(
        batch_size=train_loop_cnfg.get("batch_size"),
        prefetch_batches=1,
        dtypes={"image": torch.float32, "label": torch.int64},
    )

    # Initialize model
    model = SimpleImageClassifier(num_classes=10, lr=train_loop_cnfg.get("lr"))

    # Setup MLflow on rank 0 only to avoid duplicate logs
    rank0 = get_context().get_world_rank() == 0
    if rank0:
        worker_logger.info("🎯 Worker initialized on rank 0")
        mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)
        mlflow.pytorch.autolog(
            log_models=False,  # We will log the model manually later
        )
        mlflow.start_run(
            run_id=train_loop_cnfg.get("mlflow_run_id"),
        )
        worker_logger.info(
            f"Connected to MLflow run: [cyan]{train_loop_cnfg.get('mlflow_run_id')}[/cyan]"
        )

    # Configure and fit distributed data parallel training lightning trainer
    trainer = pl.Trainer(
        max_epochs=train_loop_cnfg.get("max_epochs"),
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=False,  # `RayTrainReportCallback` does that already
        log_every_n_steps=50,
    )
    trainer = prepare_trainer(trainer)

    # See if we have a checkpoint to resume from
    checkpoint = get_checkpoint()
    if checkpoint:
        worker_logger.info(
            f"📂 Resuming from checkpoint: [yellow]{checkpoint}[/yellow]"
        )
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir) / RayTrainReportCallback.CHECKPOINT_NAME
            trainer.fit(
                model,
                train_dataloaders=train_iter,
                val_dataloaders=val_iter,
                ckpt_path=ckpt_path,
            )
    else:
        worker_logger.info("🆕 Starting training from scratch")
        trainer.fit(model, train_dataloaders=train_iter, val_dataloaders=val_iter)

    if rank0:
        worker_logger.success("✨ Training completed on rank 0")


def train_fn_driver(train_driver_cnfg: dict) -> ray.train.Result:
    """Driver code that runs on the head node."""
    log_section("Training Pipeline", "🚀")

    # Load datasets from DVC
    data_version = train_driver_cnfg.get("data_version")
    train_ds, val_ds, data_metrics = load_data(
        version=data_version,
        limit_train=100,
        limit_val=50,  # TODO: dont hardcode in final version
    )
    # Grab a sample BEFORE training (while dataset is fresh)
    sample_batch = val_ds.take_batch(batch_size=1)
    sample_input = sample_batch["image"]  # Shape: (1, 1, 28, 28) numpy array
    logger.info(f"Input example shape: [yellow]{sample_input.shape}[/yellow]")

    ## Start MLflow run in the driver
    log_section("MLflow Configuration", "📊")
    mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)
    logger.info(f"Experiment: [cyan]{TRAINING_CONFIG.mlflow_experiment_name}[/cyan]")

    # ⚠️ IMPORTANT: Tag training rune with workflow tags
    workflow_tags = {
        "argo_workflow_uid": WORKFLOW_TAGS.argo_workflow_uid,
        "docker_image_tag": WORKFLOW_TAGS.docker_image_tag,
        "dvc_data_version": data_version,
    }

    with mlflow.start_run(
        run_name=train_driver_cnfg.get("train_loop_config").get("run_name"),
        tags=workflow_tags,
    ) as active_run:
        mlflow_run_id = active_run.info.run_id
        logger.success(f"✨ Started MLflow run: [bold cyan]{mlflow_run_id}[/bold cyan]")

        # Log DVC metadata as parameters for traceability
        mlflow.log_params(
            {
                "dvc_repo": TRAINING_CONFIG.dvc_repo,
                "dvc_train_data_path": TRAINING_CONFIG.dvc_train_data_path,
                "dvc_val_data_path": TRAINING_CONFIG.dvc_val_data_path,
                "dvc_metrics_path": TRAINING_CONFIG.dvc_metrics_path,
                "dvc_data_version": data_version,
            }
        )
        # Log data metrics from DVC as a JSON artifact
        mlflow.log_dict(data_metrics, "dvc_data_metrics.json")
        logger.info(f"Logged DVC metadata for version [bold]{data_version}[/bold]")

        # Pass run_id to workers
        train_loop_config = train_driver_cnfg.get("train_loop_config", {})
        train_loop_config["mlflow_run_id"] = mlflow_run_id

        # Configure and fit Ray TorchTrainer
        log_section("Ray TorchTrainer Configuration", "⚙️")
        num_workers = train_driver_cnfg.get("num_workers")
        logger.info(f"Number of workers: [yellow]{num_workers}[/yellow]")
        logger.info(
            f"Batch size: [yellow]{train_loop_config.get('batch_size')}[/yellow]"
        )
        logger.info(f"Learning rate: [yellow]{train_loop_config.get('lr')}[/yellow]")
        logger.info(
            f"Max epochs: [yellow]{train_loop_config.get('max_epochs')}[/yellow]"
        )

        trainer = TorchTrainer(
            train_loop_per_worker=train_fn_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=num_workers,
                use_gpu=torch.cuda.is_available()
                and torch.cuda.device_count() >= num_workers,
            ),
            # TODO: set shared temp path for distributed training
            run_config=RunConfig(
                name=train_driver_cnfg.get("train_loop_config").get(
                    "run_name", "torch_train_run"
                ),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="val_acc",
                    checkpoint_score_order="max",
                ),
                failure_config=FailureConfig(max_failures=3),
                storage_path=TRAINING_CONFIG.ray_storage_path,
            ),
            datasets={"train": train_ds, "val": val_ds},
        )

        log_section("Training Execution", "🏃")
        logger.info("Starting distributed training...")
        result = trainer.fit()

        # Log model to MLflow from best checkpoint
        if result.checkpoint:
            log_section("Model Registration", "💾")
            logger.info(
                f"Loading best checkpoint from: [cyan]{result.checkpoint}[/cyan]"
            )

            with result.checkpoint.as_directory() as checkpoint_dir:
                ckpt_path = (
                    Path(checkpoint_dir) / RayTrainReportCallback.CHECKPOINT_NAME
                )
                model = SimpleImageClassifier.load_from_checkpoint(ckpt_path)

            model = model.cpu().eval()
            logger.info(f"Logging model to MLflow run: [cyan]{mlflow_run_id}[/cyan]")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                registered_model_name=TRAINING_CONFIG.mlflow_registered_model_name,
                input_example=sample_input,
                code_paths=[
                    "src/training/model.py"
                ],  # Include model definition for unpickeling later
            )
            logger.success(
                f"✨ Model registered as [bold]{TRAINING_CONFIG.mlflow_registered_model_name}[/bold]"
            )
        else:
            logger.warning("⚠️ No checkpoint available, model not logged")

    logger.success("🎉 Training pipeline complete!")

    return result


# ============================================== #
# 🔹 SECTION: Main Entry Point
# ============================================== #
def main():
    """Main entry point for training."""
    log_section("Fashion MNIST Training", "👕")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Fashion MNIST Training (Ray Data)")
    parser.add_argument("--run-name", type=str, help="MLflow run name")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    log_section("Training Configuration", "⚙️")
    logger.info(f"Run name: {args.run_name or 'auto-generated'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Num workers: {args.num_workers}")

    log_section("CI/CD Data Contract from ENV", "📋")
    logger.info(f"Argo Workflow UID: {WORKFLOW_TAGS.argo_workflow_uid}")
    logger.info(f"Docker image tag: {WORKFLOW_TAGS.docker_image_tag}")
    logger.info(f"DVC data version: {WORKFLOW_TAGS.dvc_data_version}")

    # Build training driver configuration
    train_driver_cnfg = {
        "data_version": WORKFLOW_TAGS.dvc_data_version,
        "num_workers": args.num_workers,
        "train_loop_config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_epochs": args.max_epochs,
            "run_name": args.run_name,
        },
    }

    # Run training
    result = train_fn_driver(train_driver_cnfg)

    # Final summary
    log_section("Results", "📊")
    if result.error:
        logger.error(f"❌ Training failed with error: {result.error}")
    else:
        logger.success("✅ No errors during training")
        logger.info(f"Best checkpoint: [cyan]{result.checkpoint}[/cyan]")
        logger.info(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
