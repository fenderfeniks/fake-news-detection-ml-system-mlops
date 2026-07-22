from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
)


class NLPModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        num_classes: int = 2,
        target_precision: float | None = None,
        target_recall: float | None = None,
        class_weights=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.num_classes = num_classes
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.strict_loading = False
        if class_weights is not None:
            class_weights = list(class_weights)

        self.save_hyperparameters(ignore=["model"])

        # Базовые метрики (считаются по умолчанию с порогом 0.5 через argmax)
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Метрика для валидации (используется ТОЛЬКО для быстрого расчета порога)
        self.val_pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)

        # Метрики для финального тестирования (для построения тяжелых графиков)
        self.test_conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)
        self.test_roc_curve = MulticlassROC(num_classes=num_classes)

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits

        # Если веса заданы — используем явный CrossEntropy
        # Если нет — используем loss из модели (он без весов)
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, batch["labels"])
        else:
            loss = outputs.loss

        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits

        # Для обычных метрик используем argmax (порог 0.5)
        preds = torch.argmax(logits, dim=1)
        # Для PR-кривой нужны вероятности
        probs = torch.softmax(logits, dim=1)

        self.val_metrics.update(preds, batch["labels"])
        self.val_pr_curve.update(probs, batch["labels"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        """Быстрая математика: динамический поиск порога для целевого класса (без отрисовки)"""
        if (
            getattr(self, "target_precision", None) is not None
            or getattr(self, "target_recall", None) is not None
        ):
            precisions, recalls, thresholds = self.val_pr_curve.compute()

            p = precisions[1]
            r = recalls[1]
            t = thresholds[1]

            if self.target_precision is not None:
                valid_indices = torch.where(p >= self.target_precision)[0]
                if len(valid_indices) > 0:
                    best_idx = valid_indices[torch.argmax(r[valid_indices])]
                    best_threshold = t[best_idx].item() if best_idx < len(t) else 1.0
                    self.log("optimal_threshold_for_precision", best_threshold, logger=True)
                    self.log("expected_recall", r[best_idx].item(), logger=True)

            elif self.target_recall is not None:
                valid_indices = torch.where(r >= self.target_recall)[0]
                if len(valid_indices) > 0:
                    best_idx = valid_indices[torch.argmax(p[valid_indices])]
                    best_threshold = t[best_idx].item() if best_idx < len(t) else 0.0
                    self.log("optimal_threshold_for_recall", best_threshold, logger=True)
                    self.log("expected_precision", p[best_idx].item(), logger=True)

        self.val_pr_curve.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs.loss, outputs.logits

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Сбор данных для финальных метрик и графиков
        self.test_metrics.update(preds, batch["labels"])
        self.test_conf_matrix.update(preds, batch["labels"])
        self.test_pr_curve.update(probs, batch["labels"])
        self.test_roc_curve.update(probs, batch["labels"])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """Отрисовка финальных тяжелых графиков в конце тестирования"""

        # Если логгера нет (например, локальный дебаг), пропускаем отрисовку
        if not self.logger or not hasattr(self.logger, "experiment"):
            self.test_conf_matrix.reset()
            self.test_pr_curve.reset()
            self.test_roc_curve.reset()
            return

        mlflow_client = self.logger.experiment
        run_id = self.logger.run_id

        # 1. Отрисовка финальной Confusion Matrix
        fig_cm, ax_cm = self.test_conf_matrix.plot()
        mlflow_client.log_figure(run_id, fig_cm, "metrics_plots/final_confusion_matrix.png")
        plt.close(fig_cm)
        self.test_conf_matrix.reset()

        # 2. Отрисовка финальной PR-кривой
        fig_pr, ax_pr = self.test_pr_curve.plot()
        mlflow_client.log_figure(run_id, fig_pr, "metrics_plots/final_pr_curve.png")
        plt.close(fig_pr)
        self.test_pr_curve.reset()

        # 3. Отрисовка финальной ROC-кривой
        fig_roc, ax_roc = self.test_roc_curve.plot()
        mlflow_client.log_figure(run_id, fig_roc, "metrics_plots/final_roc_curve.png")
        plt.close(fig_roc)
        self.test_roc_curve.reset()

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = self.optimizer_cfg(params=trainable_params)

        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
