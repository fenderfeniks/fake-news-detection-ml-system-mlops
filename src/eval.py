import logging
import sys

from dotenv import load_dotenv


load_dotenv()

import hydra
from omegaconf import DictConfig

from src.utils.hydra_utils import setup_config
from src.utils.torch_utils import register_safe_globals


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    setup_config(cfg)

    logger.info("Инициализация компонентов для оценки...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()

    model_module = hydra.utils.instantiate(cfg.model_module, model=base_model)
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)
    trainer = hydra.utils.instantiate(cfg.trainer)

    ckpt_path = cfg.get("ckpt_path")

    if ckpt_path:
        logger.info(f"Загрузка кастомного PL чекпоинта из: {ckpt_path}")
        register_safe_globals()
    elif getattr(model_builder, "loaded_from_mlflow", False):
        logger.info("Модель успешно загружена из MLflow Production.")
    else:
        logger.warning("Модель не из MLflow и путь не передан. Оценка на базовых весах.")

    logger.info("Старт процесса оценки...")
    results = trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Оценка завершена.")

    # ДОБАВЛЕНО: проверка порога и сигнал для Airflow
    # trainer.test() возвращает список словарей с метриками — берём первый
    if results:
        metrics = results[0]
        # Ключ метрики — тот же что логирует твой LightningModule в test_step
        # Подставь свой: "test_f1", "test_accuracy" и т.д.
        primary_metric = metrics.get("test_f1") or metrics.get("test_accuracy")
        drift_threshold = cfg.get("drift_threshold")

        if primary_metric is not None and drift_threshold is not None:
            logger.info(f"Метрика: {primary_metric:.4f}, порог: {drift_threshold}")
            if primary_metric < drift_threshold:
                logger.error(
                    f"ДРИФТ ОБНАРУЖЕН: {primary_metric:.4f} < {drift_threshold}. "
                    "Airflow получит exit code 1 → Slack алерт сработает."
                )
                sys.exit(1)
        else:
            logger.warning(
                "Метрика или порог не найдены в конфиге/результатах — "
                "проверка дрифта пропущена. "
                f"Доступные метрики: {list(metrics.keys())}"
            )


def _check_drift(results: list[dict], drift_threshold: float, metric_key: str = "test_f1"):
    """Выделено отдельно для тестируемости без Hydra."""
    if not results:
        logger.warning("trainer.test() вернул пустые результаты — проверка дрифта пропущена.")
        return

    metrics = results[0]
    primary_metric = metrics.get(metric_key)

    if primary_metric is None:
        logger.warning(
            f"Ключ метрики '{metric_key}' не найден в результатах. "
            f"Доступные: {list(metrics.keys())}. Проверка дрифта пропущена."
        )
        return

    logger.info(f"Метрика {metric_key}: {primary_metric:.4f}, порог: {drift_threshold}")

    if primary_metric < drift_threshold:
        logger.error(f"ДРИФТ: {primary_metric:.4f} < {drift_threshold}. Выход с кодом 1.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        evaluate()
    except SystemExit as e:
        raise e  # пробрасываем дальше, не глотаем
