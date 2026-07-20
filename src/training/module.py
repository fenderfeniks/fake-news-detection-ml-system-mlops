from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class NLPModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Сохраняем гиперпараметры для логгера, исключая саму модель
        self.save_hyperparameters(ignore=["model"])

        # Метрики. Для задачи fake-news num_classes=2 подходит идеально
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # **kwargs перехватит token_type_ids и любые другие специфичные тензоры
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,  # Пробрасываем их прямо в модель HF
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        # HF модели сами считают loss, если передан labels
        loss, logits = outputs.loss, outputs.logits

        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, batch["labels"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits

        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, batch["labels"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits

        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, batch["labels"])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_cfg(params=self.model.parameters())
        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
