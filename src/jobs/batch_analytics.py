# src/jobs/batch_analytics.py
import logging
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    db_url = os.getenv("DB_CONN")
    if not db_url:
        raise ValueError(
            "Environment variable DB_CONN is not set! Check your .env or K8s variables."
        )

    logger.info("Connecting to database via DB_CONN (Mock)...")

    # 1. Загрузка данных (Заглушка адаптирована под детектирование новостей)
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

    # 2. Инициализация модели
    model_path = os.getenv("MODEL_PATH", "cointegrated/rubert-tiny2")
    device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Loading HF pipeline from {model_path} on device {device}...")
    classifier = pipeline("text-classification", model=model_path, device=device)

    # 3. Батч-инференс
    logger.info("Running batch inference...")
    results = classifier(df["text"].tolist())

    # 4. Сохранение результатов
    df["predictions"] = [res["label"] for res in results]
    df["confidence"] = [res["score"] for res in results]

    logger.info("Sample results:")
    logger.info(f"\n{df.head()}")

    logger.info("Writing results back to target database (Mock)...")
    logger.info("Batch analytics completed successfully.")


if __name__ == "__main__":
    main()
