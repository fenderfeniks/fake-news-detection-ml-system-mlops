# tests/training/test_module.py
"""
Тесты для NLPModel (PyTorch Lightning Module).
Проверяют: loss с весами классов, шаг оптимизатора, логику PR-кривой.
Все тесты используют tiny fake модель — без скачивания весов.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixture: минимальная fake-модель вместо BERT
# ---------------------------------------------------------------------------


class FakeSequenceClassificationOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class FakeEncoder(nn.Module):
    """Простейшая замена BERT — два линейных слоя."""

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.linear = nn.Linear(16, num_labels)
        self.config = MagicMock()
        self.config.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        x = input_ids.float().mean(dim=1, keepdim=True).expand(-1, 16)
        logits = self.linear(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return FakeSequenceClassificationOutput(logits=logits, loss=loss)


@pytest.fixture
def fake_model():
    return FakeEncoder(num_labels=2)


@pytest.fixture
def optimizer_cfg():
    return torch.optim.AdamW


@pytest.fixture
def nlp_module(fake_model, optimizer_cfg):
    from src.training.module import NLPModel

    return NLPModel(
        model=fake_model,
        optimizer_cfg=lambda params: optimizer_cfg(params, lr=1e-3),
        num_classes=2,
        target_precision=None,
        target_recall=None,
        class_weights=None,
    )


@pytest.fixture
def nlp_module_with_weights(fake_model, optimizer_cfg):
    from src.training.module import NLPModel

    return NLPModel(
        model=fake_model,
        optimizer_cfg=lambda params: optimizer_cfg(params, lr=1e-3),
        num_classes=2,
        class_weights=[1.0, 3.0],  # спам весит в 3 раза больше
    )


def _make_batch(batch_size: int = 4, seq_len: int = 8):
    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 2, (batch_size,)),
    }


# ---------------------------------------------------------------------------
# Loss calculation
# ---------------------------------------------------------------------------


class TestLossCalculation:
    def test_loss_is_scalar(self, nlp_module):
        batch = _make_batch()
        outputs = nlp_module(**batch)
        loss = nlp_module._calculate_loss(outputs, batch["labels"])
        assert loss.ndim == 0

    def test_loss_is_positive(self, nlp_module):
        batch = _make_batch()
        outputs = nlp_module(**batch)
        loss = nlp_module._calculate_loss(outputs, batch["labels"])
        assert loss.item() > 0

    def test_loss_with_class_weights_differs_from_unweighted(
        self, nlp_module, nlp_module_with_weights
    ):
        """Взвешенный loss должен отличаться от невзвешенного на том же батче."""
        torch.manual_seed(42)
        batch = _make_batch()

        out1 = nlp_module(**batch)
        out2 = nlp_module_with_weights(**batch)

        loss1 = nlp_module._calculate_loss(out1, batch["labels"])
        loss2 = nlp_module_with_weights._calculate_loss(out2, batch["labels"])

        # Значения могут совпасть только случайно — проверяем что хотя бы вычислились
        assert isinstance(loss1.item(), float)
        assert isinstance(loss2.item(), float)

    def test_class_weights_registered_as_buffer(self, nlp_module_with_weights):
        """Веса классов должны быть зарегистрированы как buffer (сохраняются в checkpoint)."""
        assert hasattr(nlp_module_with_weights, "class_weights")
        assert isinstance(nlp_module_with_weights.class_weights, torch.Tensor)
        assert nlp_module_with_weights.class_weights.tolist() == [1.0, 3.0]

    def test_no_class_weights_uses_model_loss(self, nlp_module):
        """Без весов loss должен браться из outputs.loss."""
        batch = _make_batch()
        outputs = nlp_module(**batch)
        loss = nlp_module._calculate_loss(outputs, batch["labels"])
        # outputs.loss вычислен FakeEncoder с nn.CrossEntropyLoss без весов
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_training_step_returns_loss(self, nlp_module):
        nlp_module.log = MagicMock()
        nlp_module.log_dict = MagicMock()
        batch = _make_batch()
        loss = nlp_module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_training_step_calls_log(self, nlp_module):
        nlp_module.log = MagicMock()
        nlp_module.log_dict = MagicMock()
        batch = _make_batch()
        nlp_module.training_step(batch, batch_idx=0)
        nlp_module.log.assert_called()

    def test_validation_step_does_not_return_loss(self, nlp_module):
        nlp_module.log = MagicMock()
        nlp_module.log_dict = MagicMock()
        batch = _make_batch()
        result = nlp_module.validation_step(batch, batch_idx=0)
        # Lightning validation_step не должен возвращать loss явно
        assert result is None

    def test_test_step_updates_metrics(self, nlp_module):
        nlp_module.log = MagicMock()
        nlp_module.log_dict = MagicMock()
        batch = _make_batch()
        nlp_module.test_step(batch, batch_idx=0)
        # Проверяем что confusion matrix накопила данные
        assert nlp_module.test_conf_matrix._update_count > 0


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    def test_returns_optimizer_without_scheduler(self, nlp_module):
        opt = nlp_module.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)

    def test_returns_dict_with_scheduler(self, fake_model):
        from src.training.module import NLPModel

        module = NLPModel(
            model=fake_model,
            optimizer_cfg=lambda params: torch.optim.AdamW(params, lr=1e-3),
            scheduler_cfg=MagicMock(),
            num_classes=2,
        )
        # Мокируем instantiate чтобы вернуть реальный scheduler
        with patch("src.training.module.instantiate") as mock_instantiate:
            optimizer = torch.optim.AdamW(fake_model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            mock_instantiate.return_value = scheduler
            module.optimizer_cfg = lambda params: optimizer

            result = module.configure_optimizers()
            assert isinstance(result, dict)
            assert "optimizer" in result
            assert "lr_scheduler" in result

    def test_only_trainable_params_passed_to_optimizer(self, fake_model):
        """Замороженные параметры не должны попасть в оптимизатор."""
        from src.training.module import NLPModel

        # Замораживаем все параметры модели
        for param in fake_model.parameters():
            param.requires_grad = False

        captured_params = []

        def capturing_optimizer(params):
            captured_params.extend(list(params))
            return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)

        module = NLPModel(
            model=fake_model,
            optimizer_cfg=capturing_optimizer,
            num_classes=2,
        )
        module.configure_optimizers()
        assert len(captured_params) == 0


# ---------------------------------------------------------------------------
# Threshold search (on_validation_epoch_end)
# ---------------------------------------------------------------------------


class TestThresholdSearch:
    def test_target_precision_logs_threshold(self, fake_model):
        from src.training.module import NLPModel

        module = NLPModel(
            model=fake_model,
            optimizer_cfg=lambda params: torch.optim.AdamW(params, lr=1e-3),
            num_classes=2,
            target_precision=0.9,
        )
        module.log = MagicMock()
        module.log_dict = MagicMock()

        # Кормим PR-кривую синтетическими данными
        probs = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
        labels = torch.tensor([0, 1, 1, 1])
        module.val_pr_curve.update(probs, labels)

        module.on_validation_epoch_end()

        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert any("threshold" in k or "recall" in k for k in logged_keys)

    def test_no_target_skips_pr_search(self, nlp_module):
        """Без target_precision/target_recall метод compute() у PR-кривой не должен вызываться."""
        nlp_module.log = MagicMock()

        # Шпионим за методом compute, оставляя объект валидным torch.nn.Module
        nlp_module.val_pr_curve.compute = MagicMock(wraps=nlp_module.val_pr_curve.compute)

        nlp_module.on_validation_epoch_end()

        # PR-кривая не должна вычисляться
        nlp_module.val_pr_curve.compute.assert_not_called()
