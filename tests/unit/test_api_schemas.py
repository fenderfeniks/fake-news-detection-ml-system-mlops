# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError

from src.api.schemas import ClassificationRequest


def test_classification_request_accepts_valid_payload():
    """Проверка создания схемы с валидным текстом."""
    body = ClassificationRequest(text="Власти опровергли слухи о закрытии метро.")

    assert body.text == "Власти опровергли слухи о закрытии метро."


def test_classification_request_rejects_missing_text():
    """Проверка, что схема падает, если не передать обязательное поле text."""
    with pytest.raises(ValidationError) as exc_info:
        ClassificationRequest()

    assert "Field required" in str(exc_info.value)
    assert "text" in str(exc_info.value)


def test_classification_request_rejects_wrong_types():
    """Проверка, что схема не принимает числа или списки вместо строки."""
    with pytest.raises(ValidationError):
        ClassificationRequest(text=["Это", "список", "а", "не", "строка"])

    with pytest.raises(ValidationError):
        ClassificationRequest(text=12345)
