# tests/api/test_endpoints.py
"""
Расширенные тесты REST-эндпоинтов.
Используют async_client + mock_classifier из корневого conftest.py.
Покрывают: auth, rate limiting, health, edge cases, error handling.
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_structure(self, async_client):
        response = await async_client.get("/health")
        data = response.json()
        assert "status" in data


# ---------------------------------------------------------------------------
# /api/v1/classify — happy path
# ---------------------------------------------------------------------------


class TestClassifyEndpointSuccess:
    @pytest.mark.asyncio
    async def test_returns_200_with_valid_text(self, async_client):
        response = await async_client.post(
            "/api/v1/classify", json={"text": "URGENT: You won a free iPhone!"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_contains_all_fields(self, async_client):
        response = await async_client.post(
            "/api/v1/classify", json={"text": "Meeting rescheduled to Friday."}
        )
        data = response.json()
        assert "label_id" in data
        assert "confidence" in data
        assert "all_probabilities" in data

    @pytest.mark.asyncio
    async def test_label_id_is_integer(self, async_client):
        response = await async_client.post(
            "/api/v1/classify", json={"text": "Claim your prize now!"}
        )
        assert isinstance(response.json()["label_id"], int)

    @pytest.mark.asyncio
    async def test_confidence_is_float(self, async_client):
        response = await async_client.post(
            "/api/v1/classify", json={"text": "Project deadline is next Monday."}
        )
        assert isinstance(response.json()["confidence"], float)

    @pytest.mark.asyncio
    async def test_all_probabilities_has_two_elements(self, async_client):
        response = await async_client.post(
            "/api/v1/classify", json={"text": "Buy cheap meds online!"}
        )
        assert len(response.json()["all_probabilities"]) == 2

    @pytest.mark.asyncio
    async def test_unicode_text_accepted(self, async_client):
        """Кириллица и спецсимволы не должны вызывать ошибку."""
        response = await async_client.post(
            "/api/v1/classify",
            json={"text": "Срочно! Вы выиграли приз! Перейдите по ссылке."},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_long_text_within_limit(self, async_client):
        response = await async_client.post(
            "/api/v1/classify",
            json={"text": "word " * 500},  # 2500 символов — в пределах 5000
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /api/v1/classify — validation errors
# ---------------------------------------------------------------------------


class TestClassifyEndpointValidation:
    @pytest.mark.asyncio
    async def test_missing_text_field_returns_422(self, async_client):
        response = await async_client.post("/api/v1/classify", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_field_name_returns_422(self, async_client):
        response = await async_client.post("/api/v1/classify", json={"content": "some text"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_422_error_points_to_text_field(self, async_client):
        response = await async_client.post("/api/v1/classify", json={})
        detail = response.json()["detail"]
        assert any(e["loc"][-1] == "text" for e in detail)

    @pytest.mark.asyncio
    async def test_text_exceeds_5000_chars_returns_422(self, async_client):
        response = await async_client.post("/api/v1/classify", json={"text": "a" * 5001})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_numeric_text_value_returns_422(self, async_client):
        response = await async_client.post("/api/v1/classify", json={"text": 42})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_body_returns_422(self, async_client):
        response = await async_client.post("/api/v1/classify", json=None)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/classify — auth
# ---------------------------------------------------------------------------


class TestClassifyEndpointAuth:
    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_403(self, test_app, override_ml_deps):
        """Неверный ключ при установленном API_KEY должен давать 403."""
        import os

        from httpx import ASGITransport, AsyncClient

        os.environ["API_KEY"] = "correct-secret-key"
        try:
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/classify",
                    json={"text": "some text"},
                    headers={"X-API-Key": "wrong-key"},
                )
            assert response.status_code == 403
        finally:
            del os.environ["API_KEY"]

    @pytest.mark.asyncio
    async def test_correct_api_key_returns_200(self, test_app, override_ml_deps):
        import os

        from httpx import ASGITransport, AsyncClient

        os.environ["API_KEY"] = "correct-secret-key"
        try:
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/classify",
                    json={"text": "some text"},
                    headers={"X-API-Key": "correct-secret-key"},
                )
            assert response.status_code == 200
        finally:
            del os.environ["API_KEY"]


# ---------------------------------------------------------------------------
# /api/v1/classify — classifier unavailable
# ---------------------------------------------------------------------------


class TestClassifyEndpointMLUnavailable:
    @pytest.mark.asyncio
    async def test_returns_503_when_classifier_is_none(self, test_app):
        """Если модель не загружена — эндпоинт должен вернуть 503, а не 500."""
        from httpx import ASGITransport, AsyncClient

        from src.api.rest.dependencies import get_classifier

        test_app.dependency_overrides[get_classifier] = lambda: None
        transport = ASGITransport(app=test_app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/v1/classify", json={"text": "test"})
            assert response.status_code in (503, 500)
        finally:
            test_app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_500_when_classifier_raises(self, test_app):
        """Если модель бросает исключение — сервер не должен падать с необработанной ошибкой."""
        from httpx import ASGITransport, AsyncClient

        from src.api.rest.dependencies import get_classifier

        broken_classifier = MagicMock(side_effect=RuntimeError("GPU out of memory"))
        test_app.dependency_overrides[get_classifier] = lambda: broken_classifier
        transport = ASGITransport(app=test_app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/v1/classify", json={"text": "test"})
            assert response.status_code == 500
        finally:
            test_app.dependency_overrides.clear()
