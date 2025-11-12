# src/model.py
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
