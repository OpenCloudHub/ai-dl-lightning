# # src/tune.py
# import argparse
# import uuid
# from pathlib import Path

# import mlflow
# import ray
# import ray.train
# import ray.train.torch
# import ray.tune
# import torch
# from ray import tune
# from ray.train import (
#     CheckpointConfig,
#     FailureConfig,
# )
# from ray.train import RunConfig as TrainRunConfig
# from ray.train import ScalingConfig as TrainScalingConfig
# from ray.train.torch import TorchTrainer
# from ray.tune import RunConfig as TuneRunConfig
# from ray.tune.integration.ray_train import TuneReportCallback

# from src._utils import setup_logging
# from src.config import BASE_CNFG
# from src.data import load_data
# from src.model import SimpleImageClassifier
# from src.train import train_fn_per_worker  # keep your worker code

# logger = setup_logging(__name__)


# def train_driver_fn(config: dict):
#     """Train driver used by Tune trials: launches a TorchTrainer per trial."""
#     # Unpack trial config
#     data_path = config.get("data_path")
#     num_workers = config.get("num_workers", 1)
#     train_loop_config = config.get("train_loop_config", {})

#     # Load datasets (vectorized normalization already applied by load_data)
#     train_ds, val_ds = load_data(data_path, limit_train=100, limit_val=50)

#     # NOTE: Pass TuneReportCallback into the *train* RunConfig (ray.train.RunConfig).
#     # This propagates ray.train.report(...) from workers to Tune.
#     trainer = TorchTrainer(
#         train_loop_per_worker=train_fn_per_worker,
#         train_loop_config=train_loop_config,
#         scaling_config=TrainScalingConfig(num_workers=num_workers),
#         run_config=TrainRunConfig(
#             name=f"train-trial_id={ray.tune.get_context().get_trial_id()}",
#             callbacks=[TuneReportCallback()],  # correct place for this callback
#             # storage_path can be set here if you want persistent checkpoint storage
#             checkpoint_config=CheckpointConfig(
#                 num_to_keep=1,
#                 checkpoint_score_attribute="val_acc",
#                 checkpoint_score_order="max",
#             ),
#             failure_config=FailureConfig(max_failures=3),
#             storage_path=BASE_CNFG.ray_storage_path,
#         ),
#         datasets={"train": train_ds, "val": val_ds},
#     )

#     # Run training (TuneReportCallback will surface reported metrics/checkpoints to Tune)
#     trainer.fit()

#     # No return value required here; Tune collects metrics via TuneReportCallback.


# def main():
#     parser = argparse.ArgumentParser(description="Fashion MNIST Hyperparameter Tuning")
#     parser.add_argument("--max-epochs", type=int, default=2)
#     parser.add_argument(
#         "--data-path", type=str, default="/workspace/project/data/FashionMNIST_parquet"
#     )
#     parser.add_argument("--num-workers", type=int, default=1)
#     args = parser.parse_args()

#     # Start parent MLflow run (optional)
#     mlflow.set_tracking_uri(BASE_CNFG.mlflow_tracking_uri)
#     mlflow.set_experiment(BASE_CNFG.mlflow_experiment_name)
#     with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
#         parent_run_id = parent_run.info.run_id
#         logger.info(f"Parent MLflow run ID: {parent_run_id}")

#         # Build Tune param space. Use grid_search if you want deterministic combos.
#         param_space = {
#             "data_path": args.data_path,
#             "num_workers": args.num_workers,
#             "train_loop_config": {
#                 "lr": tune.grid_search([1e-3]),  # explicit grid for lr
#                 "batch_size": tune.grid_search([64, 128]),  # 2 trials total
#                 "max_epochs": args.max_epochs,
#                 "parent_run_id": parent_run_id,  # optional: accessible in train_loop_config
#             },
#         }

#         tuner = tune.Tuner(
#             train_driver_fn,
#             param_space=param_space,
#             run_config=TuneRunConfig(
#                 name=f"tune_fashionmnist-{uuid.uuid4().hex[:6]}",
#                 # callbacks=[
#                 #     MLflowLoggerCallback(
#                 #         tracking_uri=BASE_CNFG.mlflow_tracking_uri,
#                 #         experiment_name=BASE_CNFG.mlflow_experiment_name,
#                 #         save_artifact=True,
#                 #     )
#                 # ],
#                 # storage_path=BASE_CNFG.ray_storage_path,  # optional
#             ),
#             # tune_config=TuneConfig(
#             #     metric="val_acc",
#             #     mode="max",
#             #     max_concurrent_trials=2,  # limit concurrency so Train runs don't fight resources
#             # ),
#         )

#         print("Starting hyperparameter tuning...")
#         result_grid = tuner.fit()

#         best_result = result_grid.get_best_result(metric="val_acc", mode="max")
#         logger.info(f"Best trial config: {best_result.config}")
#         logger.info(f"Best trial val_acc: {best_result.metrics['val_acc']:.4f}")
#         logger.info(f"Best checkpoint: {best_result.checkpoint}")

#         # Log best model to parent MLflow run: load checkpoint and log model artifact.
#         if best_result.checkpoint:
#             with best_result.checkpoint.as_directory() as checkpoint_dir:
#                 ckpt_path = (
#                     Path(checkpoint_dir) / "checkpoint"
#                 )  # adapt if yours uses different name
#                 model = SimpleImageClassifier.load_from_checkpoint(ckpt_path)

#             model = model.cpu().eval()
#             # If you want a real sample here, see persistence options below.
#             sample_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32).numpy()
#             mlflow.pytorch.log_model(
#                 pytorch_model=model,
#                 artifact_path="best_model",
#                 registered_model_name=BASE_CNFG.mlflow_registered_model_name,
#                 input_example=sample_input,
#             )

#     return result_grid


# if __name__ == "__main__":
#     main()
