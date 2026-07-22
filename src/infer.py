# src/infer.py
import logging
import os

import torch
from dotenv import load_dotenv


load_dotenv()
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils.hydra_utils import setup_config  # noqa: E402


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def infer(cfg: DictConfig) -> None:
    setup_config(cfg)

    logger.info("Загрузка токенизатора...")
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer).build()

    model_builder = hydra.utils.instantiate(cfg.model.builder, tokenizer=tokenizer)
    model = model_builder.build()
    if getattr(model_builder, "loaded_from_mlflow", False):
        logger.info(
            f"Модель загружена из MLflow Production "
            f"('{cfg.model.builder.mlflow_model_name}@{cfg.model.builder.mlflow_model_alias}')"
        )
    else:
        logger.info(f"Модель загружена с базового пути: {cfg.model.builder.model_name_or_path}")

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        logger.info(f"Подгрузка кастомных весов из: {ckpt_path}")
        if os.path.isdir(ckpt_path) and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            logger.info("Обнаружен PEFT/LoRA адаптер. Оборачиваем модель...")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ckpt_path)
        else:
            logger.warning("adapter_config.json не найден. Попытка загрузки как state_dict...")
            try:
                weight_path = (
                    os.path.join(ckpt_path, "pytorch_model.bin")
                    if os.path.isdir(ckpt_path)
                    else ckpt_path
                )
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Веса state_dict успешно загружены.")
            except Exception as e:
                logger.error(f"Не удалось загрузить чекпоинт. Ошибка: {e}")
                raise e

    model.eval()

    # Авто-определение устройства модели
    device = next(model.parameters()).device
    logger.info(f"Модель находится на устройстве: {device}")

    query = cfg.get("text", "Это пример тестовой новости для проверки пайплайна.")
    logger.info(f"Входящий запрос: {query}")

    with torch.no_grad():
        # Явно переносим батч на device модели
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # --- ИНТЕГРАЦИЯ БИЗНЕС-ЛОГИКИ ---
        threshold = cfg.get("inference", {}).get("threshold", 0.5)
        prob_fake = probabilities[0][1].item()  # Вероятность, что это фейк

        if prob_fake >= threshold:
            predicted_class_id = 1
            confidence = prob_fake
        else:
            predicted_class_id = 0
            confidence = probabilities[0][0].item()

    print("\n" + "=" * 50)
    print(f"Текст: {query}")
    print(f"Предсказанный класс ID: {predicted_class_id}")
    print(f"Уверенность модели: {confidence:.4f}")
    print(f"Используемый порог (Threshold): {threshold}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    infer()
