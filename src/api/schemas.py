# src/api/schemas.py
from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Текст новости или статьи для анализа", max_length=5000)


class ClassificationResponse(BaseModel):
    label_id: int = Field(..., description="Предсказанный класс (например, 0 - Fake, 1 - True)")
    confidence: float = Field(
        ..., description="Уверенность модели в предсказанном классе (от 0.0 до 1.0)"
    )
    all_probabilities: list[float] = Field(
        ..., description="Распределение вероятностей по всем классам"
    )
