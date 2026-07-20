# src/sdk/inference.py
import logging
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.hf_hub import download_hf_artifact


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class NLPPipeline:
    """
    SDK для инференса моделей классификации текста.
    Скрывает внутри инициализацию Hydra, токенизаторов и загрузку весов.
    """

    MODELS = {
        "fake_news": {
            "hf_repo_id": "your_username/fake-news-classifier",
            "checkpoint_path": "best.ckpt",
        },
    }

    def __init__(
        self,
        model_name: str | None = None,
        config_name: str = "main",
        checkpoint_path: str | None = None,
        hf_repo_id: str | None = None,
    ):
        if model_name and model_name in self.MODELS:
            hf_repo_id = self.MODELS[model_name]["hf_repo_id"]
            checkpoint_path = self.MODELS[model_name]["checkpoint_path"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Инициализация NLPPipeline (Classification) на устройстве: {self.device}")

        # 1. Загрузка конфигурации Hydra
        config_dir = str(Path(__file__).resolve().parents[2] / "configs")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            self.cfg = compose(config_name=config_name)
            OmegaConf.resolve(self.cfg)

        self.max_length = self.cfg.data.max_length

        # 2. Инициализация базовой архитектуры
        self.tokenizer = instantiate(self.cfg.model.tokenizer).build()
        model_builder = instantiate(self.cfg.model.builder, tokenizer=self.tokenizer)
        self.model = model_builder.build()

        # Builder сам решает, откуда взять веса: если в конфиге
        # model.builder.use_mlflow_registry: True — модель уже пришла
        # из MLflow Model Registry (Production-версия), и в общем случае
        # это финальные веса, поверх которых ничего накатывать не нужно.
        loaded_from_mlflow = getattr(model_builder, "loaded_from_mlflow", False)

        # 3. Загрузка кастомных весов (доп. LoRA/чекпоинт поверх базы).
        # Явно переданный checkpoint_path — это осознанный выбор вызывающего
        # кода, поэтому применяем его даже поверх MLflow-весов, если он
        # передан; это позволяет, например, тестировать новый LoRA-адаптер
        # поверх текущей Production-модели без переобучения с нуля.
        if checkpoint_path:
            if loaded_from_mlflow:
                logger.info(
                    "Модель загружена из MLflow Production, но также передан "
                    "checkpoint_path — накатываем его поверх Production-весов."
                )
            if hf_repo_id and not os.path.exists(checkpoint_path):
                checkpoint_path = download_hf_artifact(
                    repo_id=hf_repo_id, filename=checkpoint_path, local_dir="./models/downloaded"
                )
            self._load_weights(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, path: str) -> None:
        """Умная загрузка весов (State Dict или LoRA)."""
        logger.info(f"Загрузка кастомных весов из: {path}")

        # Проверка на PEFT/LoRA адаптер
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Обнаружен PEFT/LoRA адаптер. Оборачиваем модель...")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            logger.info("Загрузка обычного state_dict...")
            weight_path = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else path
            checkpoint = torch.load(weight_path, map_location=self.device)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean_state_dict, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

    @torch.no_grad()
    def __call__(self, texts: str | list[str]) -> list[dict[str, Any]]:
        """Логика инференса классификации."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        if hasattr(outputs, "logits"):
            probs = torch.softmax(outputs.logits, dim=-1)
        else:
            raise ValueError("Модель не вернула logits. Проверьте конфигурацию архитектуры.")

        preds = torch.argmax(probs, dim=-1)

        results = []
        for prob, pred in zip(probs, preds, strict=False):
            results.append(
                {
                    "label_id": pred.item(),
                    "confidence": round(prob[pred].item(), 4),
                    "all_probabilities": [round(p, 4) for p in prob.cpu().tolist()],
                }
            )

        return results
