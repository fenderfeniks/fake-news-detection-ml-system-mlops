# tests/api/test_schemas.py
"""
Тесты для Pydantic-схем запроса и ответа.
Чистая unit-логика — никаких HTTP-клиентов, никаких моков.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import ClassificationRequest, ClassificationResponse


class TestClassificationRequest:
    def test_valid_text(self):
        req = ClassificationRequest(text="Breaking news: scientists discover water on Mars.")
        assert req.text == "Breaking news: scientists discover water on Mars."

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ClassificationRequest()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) for e in errors)

    def test_empty_string_is_allowed(self):
        """Pydantic не запрещает пустую строку — это решение бизнес-логики."""
        req = ClassificationRequest(text="")
        assert req.text == ""

    def test_text_at_max_length(self):
        req = ClassificationRequest(text="a" * 5000)
        assert len(req.text) == 5000

    def test_text_exceeds_max_length_raises(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(text="a" * 5001)

    def test_non_string_text_raises(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(text=12345)

    def test_unicode_text_accepted(self):
        req = ClassificationRequest(text="Учёные обнаружили новый вид бактерий.")
        assert "бактерий" in req.text


class TestClassificationResponse:
    def test_valid_response(self):
        resp = ClassificationResponse(
            label_id=1,
            confidence=0.9871,
            all_probabilities=[0.0129, 0.9871],
        )
        assert resp.label_id == 1
        assert resp.confidence == pytest.approx(0.9871)
        assert len(resp.all_probabilities) == 2

    def test_label_id_zero(self):
        resp = ClassificationResponse(
            label_id=0,
            confidence=0.9954,
            all_probabilities=[0.9954, 0.0046],
        )
        assert resp.label_id == 0

    def test_missing_label_id_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResponse(
                confidence=0.9,
                all_probabilities=[0.1, 0.9],
            )

    def test_missing_confidence_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResponse(
                label_id=1,
                all_probabilities=[0.1, 0.9],
            )

    def test_missing_all_probabilities_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResponse(
                label_id=1,
                confidence=0.9,
            )

    def test_probabilities_can_be_empty_list(self):
        """Pydantic не накладывает ограничений на длину списка — это ok."""
        resp = ClassificationResponse(
            label_id=0,
            confidence=1.0,
            all_probabilities=[],
        )
        assert resp.all_probabilities == []

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_boundary_values(self, confidence):
        resp = ClassificationResponse(
            label_id=0,
            confidence=confidence,
            all_probabilities=[confidence, 1.0 - confidence],
        )
        assert resp.confidence == pytest.approx(confidence)
