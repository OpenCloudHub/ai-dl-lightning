# ==============================================================================
# Training Module
# ==============================================================================
#
# Fashion MNIST distributed training pipeline.
#
# Components:
#   - train.py: Main training script (Ray Train + MLflow)
#   - model.py: PyTorch Lightning model (ResNet18)
#   - data.py: DVC data loading with Ray Data
#   - config.py: Configuration settings
#
# Entry Point:
#   python src/training/train.py --help
#
# ==============================================================================
"""Training module for Fashion MNIST classifier."""
