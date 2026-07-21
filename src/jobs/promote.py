# src/jobs/promote.py
"""
Переставляет алиас MLflow: Staging → Production.
Запускается только через promote_to_prod DAG (ручной триггер).
"""

import logging
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI не задан.")

    model_name = os.getenv("MLFLOW_MODEL_NAME", "FakeNewsDetector")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Берём текущий Staging
    try:
        staging_mv = client.get_model_version_by_alias(model_name, "Staging")
    except Exception:
        logger.error(
            f"Алиас 'Staging' не найден для модели '{model_name}'. "
            "Убедись что retrain_model_dag отработал успешно."
        )
        sys.exit(1)

    staging_version = staging_mv.version
    staging_score = staging_mv.tags.get("val_f1", "неизвестно")

    # Проверяем что не промоутим то что уже в проде
    try:
        current_prod = client.get_model_version_by_alias(model_name, "Production")
        if current_prod.version == staging_version:
            logger.warning(f"Версия {staging_version} уже является Production. Промоут пропущен.")
            return
        prod_score = current_prod.tags.get("val_f1", "неизвестно")
        logger.info(f"Текущий Production: версия {current_prod.version} (val_f1={prod_score})")
    except Exception:
        logger.info("Production алиаса ещё нет — первый промоут.")

    # Переставляем алиас
    client.set_registered_model_alias(model_name, "Production", staging_version)
    logger.info(
        f"Версия {staging_version} (val_f1={staging_score}) переведена из Staging в Production."
    )


if __name__ == "__main__":
    main()
