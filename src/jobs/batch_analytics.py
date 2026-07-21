# src/jobs/batch_analytics.py
import logging
import os
import sys

import pandas as pd
import torch
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_mlflow(tracking_uri: str, model_name: str, alias: str = "Production"):
    """
    Загружает модель из MLflow Model Registry по алиасу.
    Возвращает HuggingFace pipeline готовый к инференсу.
    """
    import mlflow.transformers

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"
    logger.info(f"Загрузка модели из MLflow: {model_uri}")

    # mlflow.transformers.load_model возвращает готовый HF pipeline
    pipeline = mlflow.transformers.load_model(
        model_uri,
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipeline


def main():
    # 1. Проверка обязательных переменных окружения
    db_url = os.getenv("DB_CONN")
    if not db_url:
        raise ValueError("DB_CONN is not set! Check your K8s secrets.")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set! Check your configmap.")

    model_name = os.getenv("MLFLOW_MODEL_NAME", "FakeNewsDetector")

    # 2. Загрузка модели
    try:
        classifier = load_model_from_mlflow(mlflow_uri, model_name)
    except Exception as e:
        logger.exception(f"Не удалось загрузить модель из MLflow: {e}")
        sys.exit(1)

    # 3. Загрузка данных из БД (заглушка — замени на реальный коннектор)
    logger.info("Connecting to database via DB_CONN...")
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "Ученые доказали, что Земля плоская и стоит на трех китах.",
                "Центробанк объявил о снижении ключевой ставки на 1 процентный пункт.",
                "Шок! Инопланетяне похитили мэра Нью-Йорка во время пресс-конференции!",
            ],
        }
    )
    logger.info(f"Loaded {len(df)} records for batch inference.")

    # 4. Батч-инференс
    logger.info("Running batch inference...")
    results = classifier(df["text"].tolist())

    # 5. Сохранение результатов
    df["prediction"] = [res["label"] for res in results]
    df["confidence"] = [round(res["score"], 4) for res in results]

    logger.info(f"Sample results:\n{df.head()}")
    logger.info("Writing results back to database (Mock)...")
    logger.info("Batch analytics completed successfully.")


if __name__ == "__main__":
    main()
