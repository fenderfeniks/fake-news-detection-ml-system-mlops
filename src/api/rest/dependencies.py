"""
Dependency Injection (DI) контейнер для FastAPI.
Обеспечивает безопасный доступ к ML-моделям и сервисам для всех эндпоинтов.
"""

from fastapi import HTTPException, Request

from src.sdk.inference import NLPPipeline


def get_classifier(request: Request) -> NLPPipeline:
    """Достает модель классификации из глобальной памяти сервера."""
    classifier = request.app.state.ml_models.get("classifier")
    if not classifier:
        raise HTTPException(
            status_code=503, detail="Модель классификации еще не загружена в память."
        )
    return classifier
