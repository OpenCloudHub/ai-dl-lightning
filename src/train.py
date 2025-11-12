import argparse
from pathlib import Path

import lightning.pytorch as pl
import mlflow
import ray
import torch
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from ray.train import (
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
    get_checkpoint,
    get_context,
    get_dataset_shard,
    report,
)
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from src._utils import setup_logging
from src.config import BASE_CNFG
from src.data import load_data
from src.model import SimpleImageClassifier

logger = setup_logging(__name__)


# ========== Training Functions ========== #
def train_fn_per_worker(train_loop_cnfg: dict):
    """Training code that runs on each worker."""
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

    # Setup MLflow logger on rank 0 only to avoid duplicate logs
    # We will add tags to connect models and workflow runs later
    rank0 = get_context().get_world_rank() == 0
    if rank0:
        mlflow_logger = MLFlowLogger(
            # run_id=mlflow_run_id,
            experiment_name=BASE_CNFG.mlflow_experiment_name,
            tracking_uri=BASE_CNFG.mlflow_tracking_uri,
            tags={
                "ray_experiment_name": get_context().get_experiment_name(),
                "argo_workflow_uid": BASE_CNFG.argo_workflow_uid,
            },
        )
    else:
        mlflow_logger = None  # Disable logging on other ranks

    # Configure and fit distributed data parallel training lightning trainer
    trainer = pl.Trainer(
        max_epochs=train_loop_cnfg.get("max_epochs"),
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        logger=mlflow_logger,
        enable_checkpointing=False,  # `RayTrainReportCallback` does that already
        log_every_n_steps=50,
    )
    trainer = prepare_trainer(trainer)
    checkpoint = get_checkpoint()
    if checkpoint:
        logger.info(f"Resuming from checkpoint at {checkpoint}.")
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir) / RayTrainReportCallback.CHECKPOINT_NAME
            trainer.fit(
                model,
                train_dataloaders=train_iter,
                val_dataloaders=val_iter,
                ckpt_path=ckpt_path,
            )
    else:
        trainer.fit(model, train_dataloaders=train_iter, val_dataloaders=val_iter)

    # Report MLflow run_id back to Ray (only rank 0 has it)
    if rank0 and mlflow_logger:
        final_metrics = {
            k: float(v.item()) if hasattr(v, "item") else float(v)
            for k, v in trainer.callback_metrics.items()
        }
        final_metrics["mlflow_run_id"] = mlflow_logger.run_id
        report(final_metrics, checkpoint=get_checkpoint())


def train_fn_driver(train_driver_cnfg: dict) -> ray.train.Result:
    """Driver code that runs on the head node."""

    # Load datasets from parquet files
    train_ds, val_ds = load_data(
        data_path=train_driver_cnfg.get("data_path"), limit_train=100, limit_val=50
    )
    # Grab a sample BEFORE training (while dataset is fresh)
    sample_batch = val_ds.take_batch(batch_size=1)
    sample_input = sample_batch["image"]  # Shape: (1, 1, 28, 28) numpy array

    # Configure and fit Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_fn_per_worker,
        train_loop_config=train_driver_cnfg.get("train_loop_config"),
        scaling_config=ScalingConfig(
            num_workers=train_driver_cnfg.get("num_workers"),
            use_gpu=torch.cuda.is_available()
            and torch.cuda.device_count() >= train_driver_cnfg.get("num_workers"),
        ),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_acc",
                checkpoint_score_order="max",
            ),
            failure_config=FailureConfig(max_failures=3),
            storage_path=BASE_CNFG.ray_storage_path,
        ),
        datasets={"train": train_ds, "val": val_ds},
    )
    result = trainer.fit()

    # Attach sample input to result for later use
    result._sample_input = sample_input

    return result


# ========= Main Entry Point ========== #
def main():
    """Main entry point for training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fashion MNIST Training (Ray Data)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument(
        "--data-path", type=str, default="/workspace/project/data/FashionMNIST_parquet"
    )
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    # Build training driver configuration
    train_driver_cnfg = {
        "data_path": args.data_path,
        "num_workers": args.num_workers,
        "train_loop_config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_epochs": args.max_epochs,
        },
    }

    # Run training
    result = train_fn_driver(train_driver_cnfg)

    logger.info("✅ Training complete!")
    logger.info(f"Metrics: {result.metrics}")
    logger.info(f"Best checkpoint: {result.checkpoint}")
    if result.error:
        logger.info(f"Error: {result.error}")

    # Log model to MLflow from best checkpoint
    mlflow_run_id = result.metrics.get("mlflow_run_id")
    if mlflow_run_id and result.checkpoint:
        logger.info(f"Logging best model to MLflow run: {mlflow_run_id}")

        # Load model
        with result.checkpoint.as_directory() as checkpoint_dir:
            ckpt_path = Path(checkpoint_dir) / RayTrainReportCallback.CHECKPOINT_NAME
            model = SimpleImageClassifier.load_from_checkpoint(ckpt_path)

        # Move model to CPU for logging (serving will use CPU/GPU as needed)
        model = model.cpu().eval()

        mlflow.set_tracking_uri(BASE_CNFG.mlflow_tracking_uri)
        with mlflow.start_run(run_id=mlflow_run_id):
            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                registered_model_name=BASE_CNFG.mlflow_registered_model_name,
                input_example=result._sample_input,
            )

            logger.info("✅ Model logged to MLflow!")


if __name__ == "__main__":
    main()
