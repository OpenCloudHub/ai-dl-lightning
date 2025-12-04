# ==============================================================================
# Fashion MNIST Model Definition
# ==============================================================================
#
# PyTorch Lightning module for Fashion MNIST classification.
#
# Model Architecture:
#   - Base: ResNet18 (pretrained weights NOT used - trained from scratch)
#   - Modified: First conv layer adapted for grayscale (1 channel instead of 3)
#   - Output: 10 classes (Fashion MNIST categories)
#
# Training Features:
#   - AdamW optimizer with weight decay
#   - Cross-entropy loss
#   - Comprehensive metrics: Accuracy, Precision, Recall, F1 (multiclass)
#
# Fashion MNIST Classes:
#   0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
#   5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
#
# Usage:
#   model = SimpleImageClassifier(num_classes=10, lr=1e-3)
#   trainer = pl.Trainer(max_epochs=10)
#   trainer.fit(model, train_dataloader, val_dataloader)
#
# Note:
#   This model expects input batches as dicts: {'image': tensor, 'label': tensor}
#   Designed for use with Ray Data iterators (not standard PyTorch DataLoaders)
#
# ==============================================================================

from typing import Dict, Tuple

import lightning.pytorch as pl
import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models import resnet18


class SimpleImageClassifier(pl.LightningModule):
    """Simple Image Classifier using ResNet18 adapted for grayscale images."""

    def __init__(self, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "precision": MulticlassPrecision(num_classes=num_classes),
                "recall": MulticlassRecall(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared logic for train/val steps."""
        x = batch["image"]
        y = batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)

        metrics = self.train_metrics(preds, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch)

        metrics = self.val_metrics(preds, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
