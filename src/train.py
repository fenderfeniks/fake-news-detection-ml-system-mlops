"""
Главный скрипт запуска обучения (Orchestrator).
Инициализирует Hydra, настраивает воспроизводимость, собирает компоненты
и запускает тренировочный цикл PyTorch Lightning.
"""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig


load_dotenv()


from src.utils.hydra_utils import setup_config  # noqa: E402
from src.utils.mlflow_requirements import get_inference_pip_requirements  # noqa: E402
from src.utils.torch_utils import register_safe_globals  # noqa: E402


# Настройка логгера для текущего файла
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Основная функция запуска эксперимента.

    Args:
        cfg (DictConfig): Разрешенная конфигурация Hydra.
    """

    # ИСПРАВЛЕНИЕ: Перехватываем возвращаемый валидированный конфиг
    cfg = setup_config(cfg)

    # 2. Обеспечение воспроизводимости
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Зафиксирован глобальный seed: {cfg.seed}")

    # 3. Сборка токенизатора через нашу фабрику
    logger.info("Инициализация токенизатора...")
    tokenizer_builder = hydra.utils.instantiate(cfg.model.tokenizer)
    tokenizer = tokenizer_builder.build()

    # 4. Сборка базовой модели
    logger.info("Загрузка и сборка архитектуры модели...")
    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    base_model = model_builder.build()

    # 5. Инициализация DataModule
    logger.info("Инициализация DataModule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)

    # ИСПРАВЛЕНО: сначала prepare_data (скачивает и кэширует датасет),
    # потом setup (загружает из кэша и вычисляет class_weights)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    class_weights = getattr(datamodule, "class_weights", None)
    if class_weights is not None:
        logger.info(f"Веса классов из DataModule: {class_weights.tolist()}")

    # 6. Инициализация LightningModule — ПОСЛЕ DataModule, передаём веса
    logger.info("Инициализация PyTorch Lightning Module...")
    model_module = hydra.utils.instantiate(
        cfg.model_module,
        model=base_model,
        class_weights=class_weights.tolist() if class_weights is not None else None,
    )

    # 7. Сборка PyTorch Lightning Trainer (номера сдвинулись)
    logger.info("Инициализация PyTorch Lightning Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 8. Запуск обучения
    logger.info("Старт тренировочного цикла...")
    try:
        trainer.fit(model=model_module, datamodule=datamodule)
        logger.info("Обучение успешно завершено!")

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            # === ЭЛЕГАНТНОЕ РЕШЕНИЕ (0 RAM OVERHEAD) ===
            # Загружаем лучшие веса напрямую в существующий объект модели.
            # Тензоры просто перезапишутся in-place.
            register_safe_globals()
            logger.info("Загрузка лучших весов в память...")
            checkpoint = torch.load(
                best_ckpt_path, map_location=model_module.device, weights_only=False
            )
            model_module.load_state_dict(checkpoint["state_dict"])

            # 9. Финальное тестирование на лучших весах
            logger.info("Запуск тестирования на отложенной выборке...")
            trainer.test(model=model_module, datamodule=datamodule)

            # --- ИНТЕГРАЦИЯ С MLFLOW MODEL REGISTRY ---
            # Поскольку все вычисления и тесты завершены, мы можем безопасно
            # мутировать текущую модель (впекать LoRA), не боясь ничего сломать.
            import mlflow.transformers
            from mlflow.tracking import MlflowClient

            logger.info("Извлекаем и регистрируем модель в MLflow...")
            register_safe_globals()

            model_to_save = model_module.model
            best_score = trainer.checkpoint_callback.best_model_score
            best_score = float(best_score) if best_score is not None else None

            # Впекаем LoRA (операция in-place)
            if hasattr(model_to_save, "merge_and_unload"):
                logger.info("Обнаружена LoRA. Впекаем адаптеры...")
                model_to_save = model_to_save.merge_and_unload()

            # 3. Логируем и регистрируем модель нативным MLflow flavor
            #    (вместо ручного save_pretrained + log_artifacts) —
            #    так модель хранится вместе с сигнатурой и грузится
            #    единым вызовом mlflow.transformers.load_model(...).
            #
            #    pip_requirements передаём явно, читая их из группы
            #    [project.optional-dependencies.inference-core] в
            #    pyproject.toml — единственный источник правды о том,
            #    что реально нужно для инференса модели. Без этого
            #    MLflow пытается определить зависимости автоматически
            #    (get_default_pip_requirements) и может потребовать
            #    пакеты вроде torchvision, даже если они не используются.
            reg_model_name = cfg.model.builder.get("mlflow_model_name", "FakeNewsDetector")
            pyproject_path = Path(cfg.paths.root_dir) / "pyproject.toml"
            pip_requirements = get_inference_pip_requirements(pyproject_path)

            mlflow.set_tracking_uri(cfg.logger.tracking_uri)

            model_info = mlflow.transformers.log_model(
                transformers_model={"model": model_to_save, "tokenizer": tokenizer},
                name="model",
                task="text-classification",
                registered_model_name=reg_model_name,
                pip_requirements=pip_requirements or None,
            )
            mv_version = model_info.registered_model_version
            logger.info(
                f"Модель зарегистрирована в реестре под именем '{reg_model_name}', "
                f"Версия: {mv_version}"
            )

            if best_score is not None:
                mlflow.log_metric("promotion_candidate_score", best_score)

            # 4. Присваиваем алиас Production, только если новая версия
            #    не хуже текущей Production (или Production ещё нет).
            #    Обходим случайный откат метрики из-за нестабильного рана.
            client = MlflowClient()
            client.set_registered_model_alias(
                name=reg_model_name, alias="Staging", version=mv_version
            )
            if best_score is not None:
                client.set_model_version_tag(reg_model_name, mv_version, "val_f1", str(best_score))
            logger.info(
                f"Версия {mv_version} помечена как Staging. Запусти promote_to_prod для продакшна."
            )

    except Exception as e:
        logger.exception("Произошла критическая ошибка во время обучения:")
        raise e


if __name__ == "__main__":
    train()
