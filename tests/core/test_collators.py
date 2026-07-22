# tests/core/test_collators.py
"""
Тесты для DynamicTextCollator.
Используем реальный быстрый токенизатор (bert-base-uncased) через
pytest.importorskip — тест пропустится если transformers не установлен.
"""

import pytest
import torch


transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import AutoTokenizer  # noqa: E402

from src.core.data.collators import DynamicTextCollator  # noqa: E402


TOKENIZER_NAME = "bert-base-uncased"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture
def collator(tokenizer):
    return DynamicTextCollator(
        tokenizer=tokenizer,
        max_length=64,
        text_column="text",
        target_column="label",
    )


class TestDynamicTextCollator:
    def test_returns_input_ids_and_attention_mask(self, collator):
        batch = [{"text": "Hello world", "label": 0}]
        result = collator(batch)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_returns_labels_when_present(self, collator):
        batch = [
            {"text": "Spam email here", "label": 1},
            {"text": "Normal message", "label": 0},
        ]
        result = collator(batch)
        assert "labels" in result
        assert result["labels"].tolist() == [1, 0]

    def test_no_labels_when_target_column_missing(self, tokenizer):
        """Инференс-батч без таргетов не должен падать."""
        collator = DynamicTextCollator(
            tokenizer=tokenizer,
            max_length=64,
            text_column="text",
            target_column="label",
        )
        batch = [{"text": "Just text, no label"}]
        result = collator(batch)
        assert "labels" not in result

    def test_dynamic_padding_matches_longest_sequence(self, collator):
        """Все последовательности в батче паддятся до длины самой длинной."""
        batch = [
            {"text": "Hi", "label": 0},
            {"text": "This is a much longer sentence for testing", "label": 1},
        ]
        result = collator(batch)
        seq_len = result["input_ids"].shape[1]
        # Все строки должны иметь одинаковую длину после паддинга
        assert result["input_ids"].shape == (2, seq_len)
        assert result["attention_mask"].shape == (2, seq_len)

    def test_truncation_at_max_length(self, tokenizer):
        """Тексты длиннее max_length должны усекаться."""
        collator = DynamicTextCollator(
            tokenizer=tokenizer,
            max_length=16,
            text_column="text",
            target_column="label",
        )
        long_text = "word " * 200
        batch = [{"text": long_text, "label": 0}]
        result = collator(batch)
        assert result["input_ids"].shape[1] <= 16

    def test_output_tensors_have_correct_dtype(self, collator):
        batch = [{"text": "Test", "label": 1}]
        result = collator(batch)
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_batch_size_preserved(self, collator):
        batch = [
            {"text": "First", "label": 0},
            {"text": "Second", "label": 1},
            {"text": "Third", "label": 0},
        ]
        result = collator(batch)
        assert result["input_ids"].shape[0] == 3

    def test_custom_column_names(self, tokenizer):
        collator = DynamicTextCollator(
            tokenizer=tokenizer,
            max_length=32,
            text_column="content",
            target_column="target",
        )
        batch = [{"content": "Some email text", "target": 1}]
        result = collator(batch)
        assert "input_ids" in result
        assert result["labels"].tolist() == [1]
