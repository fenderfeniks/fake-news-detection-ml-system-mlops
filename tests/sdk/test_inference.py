# tests/sdk/test_inference.py
"""
Тесты для NLPPipeline — inference SDK.
Все тяжёлые зависимости (torch, transformers, mlflow, hydra) мокируются,
поэтому тесты запускаются быстро и без GPU/сети.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_outputs(logits: list[list[float]]):
    """Создаёт mock HuggingFace ModelOutput с нужными логитами."""
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor(logits, dtype=torch.float)
    return mock_outputs


def _make_pipeline_with_mock_model(logits: list[list[float]], threshold: float = 0.5):
    """
    Возвращает NLPPipeline с замоканными: Hydra cfg, tokenizer, model.
    Позволяет тестировать бизнес-логику __call__ без реального BERT.
    """
    from src.sdk.inference import NLPPipeline

    mock_cfg = MagicMock()
    mock_cfg.data.max_length = 256
    mock_cfg.get.return_value = {"threshold": threshold}

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones(len(logits), 4, dtype=torch.long),
        "attention_mask": torch.ones(len(logits), 4, dtype=torch.long),
    }
    # Имитируем .to(device)
    mock_tokenizer.return_value = MagicMock()
    mock_tokenizer.return_value.__getitem__ = lambda self, k: torch.ones(len(logits), 4, dtype=torch.long)
    tokenized = MagicMock()
    tokenized.to.return_value = tokenized
    tokenized.__iter__ = lambda self: iter({})
    mock_tokenizer.return_value = tokenized

    mock_model = MagicMock()
    mock_model.return_value = _make_mock_outputs(logits)
    mock_model.parameters.return_value = []

    pipeline = object.__new__(NLPPipeline)
    pipeline.cfg = mock_cfg
    pipeline.tokenizer = mock_tokenizer
    pipeline.model = mock_model
    pipeline.device = torch.device("cpu")
    pipeline.max_length = 256
    return pipeline


# ---------------------------------------------------------------------------
# __call__ output structure
# ---------------------------------------------------------------------------


class TestNLPPipelineCall:
    def test_single_string_input_returns_list(self):
        """Строка должна оборачиваться в список автоматически."""
        from src.sdk.inference import NLPPipeline

        pipeline = _make_pipeline_with_mock_model([[0.1, 0.9]])

        # Патчим токенизатор чтобы вернуть правильный mock
        with patch.object(pipeline, "tokenizer") as mock_tok:
            tokenized = MagicMock()
            tokenized.to.return_value = tokenized
            mock_tok.return_value = tokenized

            with patch.object(pipeline, "model") as mock_model:
                mock_model.return_value = _make_mock_outputs([[0.1, 0.9]])

                result = pipeline("Free iPhone click here")
                assert isinstance(result, list)
                assert len(result) == 1

    def test_result_contains_required_keys(self):
        from src.sdk.inference import NLPPipeline

        pipeline = _make_pipeline_with_mock_model([[0.2, 0.8]])

        with patch.object(pipeline, "tokenizer") as mock_tok, \
             patch.object(pipeline, "model") as mock_model:
            tokenized = MagicMock()
            tokenized.to.return_value = tokenized
            mock_tok.return_value = tokenized
            mock_model.return_value = _make_mock_outputs([[0.2, 0.8]])

            result = pipeline("Test text")
            assert "label_id" in result[0]
            assert "confidence" in result[0]
            assert "all_probabilities" in result[0]

    def test_all_probabilities_length_equals_num_classes(self):
        from src.sdk.inference import NLPPipeline

        pipeline = _make_pipeline_with_mock_model([[0.3, 0.7]])

        with patch.object(pipeline, "tokenizer") as mock_tok, \
             patch.object(pipeline, "model") as mock_model:
            tokenized = MagicMock()
            tokenized.to.return_value = tokenized
            mock_tok.return_value = tokenized
            mock_model.return_value = _make_mock_outputs([[0.3, 0.7]])

            result = pipeline("Test")
            assert len(result[0]["all_probabilities"]) == 2


# ---------------------------------------------------------------------------
# Threshold logic (без реального pipeline — чистая unit-логика)
# ---------------------------------------------------------------------------


class TestThresholdLogic:
    """
    Тестируем логику применения порога напрямую через softmax + threshold,
    повторяя то что делает NLPPipeline.__call__.
    """

    def _apply_threshold(self, logits: list[float], threshold: float) -> dict:
        """Реплика логики из NLPPipeline.__call__."""
        prob = torch.softmax(torch.tensor(logits), dim=0)
        prob_spam = prob[1].item()
        if prob_spam >= threshold:
            label_id = 1
            confidence = prob_spam
        else:
            label_id = 0
            confidence = prob[0].item()
        return {
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "all_probabilities": [round(p, 4) for p in prob.tolist()],
        }

    def test_high_spam_probability_gives_label_1(self):
        result = self._apply_threshold([0.1, 2.5], threshold=0.5)
        assert result["label_id"] == 1

    def test_low_spam_probability_gives_label_0(self):
        result = self._apply_threshold([2.5, 0.1], threshold=0.5)
        assert result["label_id"] == 0

    def test_exactly_at_threshold_is_spam(self):
        """Значение ровно на пороге должно классифицироваться как спам."""
        # Подбираем логиты чтобы softmax дал ровно 0.5
        result = self._apply_threshold([0.0, 0.0], threshold=0.5)
        assert result["label_id"] == 1  # 0.5 >= 0.5

    def test_high_threshold_makes_classifier_conservative(self):
        """При высоком пороге (0.95) даже уверенный спам может пройти как ham."""
        result = self._apply_threshold([0.3, 1.5], threshold=0.95)
        # softmax([0.3, 1.5]) ≈ [0.225, 0.775] — меньше 0.95, должен быть ham
        assert result["label_id"] == 0

    def test_confidence_is_rounded_to_4_decimals(self):
        result = self._apply_threshold([0.1, 2.5], threshold=0.5)
        assert len(str(result["confidence"]).split(".")[-1]) <= 4

    def test_probabilities_sum_to_one(self):
        result = self._apply_threshold([1.0, 2.0], threshold=0.5)
        total = sum(result["all_probabilities"])
        assert abs(total - 1.0) < 1e-4

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.5422, 0.7, 0.9])
    def test_various_thresholds_do_not_raise(self, threshold):
        result = self._apply_threshold([1.0, 1.0], threshold=threshold)
        assert result["label_id"] in (0, 1)
