# tests/conftest.py
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# Настройка окружения
os.environ["PROJECT_ROOT"] = os.getcwd()
os.environ.setdefault("PROJECT_NAME", "Test Fake News API")
os.environ.setdefault("PROJECT_VERSION", "0.1.0")
os.environ.setdefault("PROJECT_DESCRIPTION", "Test API")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test_token")
os.environ.setdefault("TG_BOT_TOKEN", "test_bot_token")

# Импортируем нашу новую зависимость
from src.api.rest.dependencies import get_classifier
from src.api.rest.server import create_app


@pytest.fixture(scope="function")
def test_app():
    """Фабрика: создает чистое приложение для каждого теста без загрузки моделей."""
    app = create_app(load_ml=False)
    app.state.ml_models = {}
    return app


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Мок классификатора, имитирующий возврат от NLPPipeline.__call__"""
    classifier = MagicMock(name="NLPPipeline")
    # При вызове возвращаем структуру, которую ожидает эндпоинт
    classifier.return_value = [
        {"label_id": 1, "confidence": 0.9850, "all_probabilities": [0.0150, 0.9850]}
    ]
    return classifier


@pytest.fixture
def override_ml_deps(test_app, mock_classifier):
    """Переопределяем зависимости для экземпляра приложения test_app."""
    test_app.dependency_overrides[get_classifier] = lambda: mock_classifier

    # Обновляем state
    test_app.state.ml_models = {
        "classifier": mock_classifier,
    }

    yield test_app
    test_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_app, override_ml_deps):
    """Клиент использует свежий test_app с переопределенными зависимостями."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
