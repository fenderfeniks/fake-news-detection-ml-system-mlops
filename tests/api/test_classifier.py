# tests/api/test_classifier.py
import pytest


@pytest.mark.asyncio
async def test_classify_endpoint_success(async_client):
    """Проверка успешной классификации текста."""
    payload = {"text": "Ученые открыли новый вид бактерий на дне Марианской впадины."}

    response = await async_client.post("/api/v1/classify", json=payload)

    # Проверяем статусы и структуру ответа
    assert response.status_code == 200

    data = response.json()

    # Опираемся на структуру, которую возвращает mock_classifier из conftest.py
    assert "label_id" in data
    assert "confidence" in data
    assert "all_probabilities" in data

    # Так как мы замокали ответ, мы точно знаем, какие значения должны вернуться
    assert data["label_id"] == 1
    assert data["confidence"] == 0.9850
    assert len(data["all_probabilities"]) == 2


@pytest.mark.asyncio
async def test_classify_endpoint_validation_error(async_client):
    """Проверка 422 ошибки при неверном payload (например, старый RAG формат)."""
    payload = {
        "query": "Как перезапустить pod?",  # Поле называется query, а не text
        "use_rag": True,
    }

    response = await async_client.post("/api/v1/classify", json=payload)

    assert response.status_code == 422

    # Проверяем, что FastAPI четко указал, где ошибка
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "text"]
    assert detail[0]["msg"] == "Field required"


@pytest.mark.asyncio
async def test_classify_endpoint_empty_body(async_client):
    """Проверка обработки абсолютно пустого тела запроса."""
    response = await async_client.post("/api/v1/classify", json={})

    assert response.status_code == 422
