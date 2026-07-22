# src/jobs/promote.py
"""
Переставляет алиас MLflow: Staging → Production.
Запускается локально или через promote_to_prod DAG.
Содержит гейт проверки метрик: промоутит только если новая модель лучше.
"""

import logging
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Для локальной разработки удобно подгружать .env (в K8s это проигнорируется)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise OSError("MLFLOW_TRACKING_URI не задан.")

    model_name = os.getenv("MLFLOW_MODEL_NAME", "FakeNewsDetector")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1. Берём текущий Staging
    try:
        staging_mv = client.get_model_version_by_alias(model_name, "Staging")
    except Exception:
        logger.error(
            f"Алиас 'Staging' не найден для модели '{model_name}'. "
            "Убедись что retrain_model_dag отработал успешно."
        )
        sys.exit(1)

    staging_version = staging_mv.version
    staging_score_str = staging_mv.tags.get("val_f1")

    if staging_score_str is None:
        logger.error("У Staging модели нет тега 'val_f1'. Невозможно оценить качество.")
        sys.exit(1)

    staging_score = float(staging_score_str)

    # 2. Проверяем текущий Production
    try:
        current_prod = client.get_model_version_by_alias(model_name, "Production")
        if current_prod.version == staging_version:
            logger.warning(f"Версия {staging_version} уже является Production. Промоут пропущен.")
            return

        prod_score_str = current_prod.tags.get("val_f1")
        prod_score = float(prod_score_str) if prod_score_str else 0.0

        logger.info(f"Текущий Production: версия {current_prod.version} (val_f1={prod_score:.4f})")
    except Exception:
        logger.info("Production алиаса ещё нет — первый промоут.")
        prod_score = 0.0

    # 3. Сравнение метрик (Gate)
    logger.info(f"Сравнение: Staging ({staging_score:.4f}) vs Production ({prod_score:.4f})")

    if staging_score > prod_score:
        client.set_registered_model_alias(model_name, "Production", staging_version)
        logger.info(
            f"УСПЕХ! Версия {staging_version} (val_f1={staging_score:.4f}) "
            "стала новой Production моделью."
        )
    else:
        logger.warning(
            f"ОТКАЗ: Модель в Staging ({staging_score:.4f}) не превосходит "
            f"текущую Production ({prod_score:.4f}). Промоут отменен."
        )


if __name__ == "__main__":
    main()
