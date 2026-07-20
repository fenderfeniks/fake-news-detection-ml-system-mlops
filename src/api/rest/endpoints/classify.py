import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.metrics import CLASSIFICATION_INFERENCE_TIME, CLASSIFICATION_REQUESTS_TOTAL
from src.api.rest.dependencies import get_classifier
from src.api.rest.limiter import limiter
from src.api.schemas import ClassificationRequest, ClassificationResponse
from src.sdk.inference import NLPPipeline


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Classification"])


@router.post("/classify", response_model=ClassificationResponse)
@limiter.limit("20/minute")
async def classify_text(
    request: Request,
    body: ClassificationRequest,
    classifier: NLPPipeline = Depends(get_classifier),
):
    try:
        CLASSIFICATION_REQUESTS_TOTAL.labels(source="rest").inc()

        # Запускаем инференс в отдельном потоке (to_thread), чтобы не блокировать Event Loop
        with CLASSIFICATION_INFERENCE_TIME.labels(source="rest").time():
            results = await asyncio.to_thread(classifier, body.text)

        # Достаем результат для первого (и единственного) текста в батче
        res = results[0]

        return ClassificationResponse(
            label_id=res["label_id"],
            confidence=res["confidence"],
            all_probabilities=res["all_probabilities"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка инференса: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка классификации текста.") from e
