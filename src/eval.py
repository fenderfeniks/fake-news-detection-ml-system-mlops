import logging
import os
import sys

from dotenv import load_dotenv


load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.torch_utils import register_safe_globals  # noqa: E402


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

    # Умная загрузка весов
    if ckpt_path:
        logger.info(f"Загрузка кастомного чекпоинта из: {ckpt_path}")
        register_safe_globals()

        # Если это LoRA папкa или raw state_dict, загружаем in-place до запуска trainer.test
        if os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            from peft import PeftModel

            model_module.model = PeftModel.from_pretrained(model_module.model, ckpt_path)
            ckpt_path = None  # Сбрасываем, так как веса уже в памяти

    elif getattr(model_builder, "loaded_from_mlflow", False):
        logger.info("Модель успешно загружена из MLflow Production.")
    else:
        logger.warning("Модель не из MLflow и путь не передан. Оценка на базовых весах.")

    logger.info("Старт процесса оценки...")
    results = trainer.test(model=model_module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Оценка завершена.")

    # Используем выделенную функцию
    drift_threshold = cfg.get("drift_threshold")
    metric_key = cfg.get("drift_metric_key", "test_f1")

    if drift_threshold is not None:
        _check_drift(results, drift_threshold=drift_threshold, metric_key=metric_key)


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
