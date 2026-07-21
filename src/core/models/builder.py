# src/core/model/builder.py
import importlib
import logging
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HFModelBuilder:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        auto_model_class: str = "transformers.AutoModelForSequenceClassification",
        cache_dir: Optional[str] = None,
        quantization_config: Optional[Any] = None,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        peft_config: Optional[Any] = None,
        num_labels: int = 2,
        finetuning_type: str = "full",
        use_mlflow_registry: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_model_alias: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.auto_model_class = auto_model_class
        self.cache_dir = cache_dir
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.peft_config = peft_config
        self.num_labels = num_labels
        self.finetuning_type = finetuning_type

        # --- Параметры MLflow ---
        self.use_mlflow_registry = use_mlflow_registry
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_alias = mlflow_model_alias

        # Флаг, отражающий реальный источник весов после build().
        # NLPPipeline проверяет его, чтобы понять, применять ли поверх
        # доп. state_dict/LoRA чекпоинт из checkpoint_path/hf_repo_id.
        self.loaded_from_mlflow: bool = False

    def _resolve_load_path(self) -> str:
        """
        Определяет, откуда грузить веса: из MLflow Model Registry
        или с исходного model_name_or_path.
        """
        if not (self.use_mlflow_registry and self.mlflow_model_name):
            return self.model_name_or_path

        import mlflow
        import os

        model_uri = f"models:/{self.mlflow_model_name}@{self.mlflow_model_alias}"
        logger.info(f"Запрос модели '{model_uri}' из MLflow Model Registry...")
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            downloaded_path = mlflow.artifacts.download_artifacts(model_uri)
            logger.info(f"Артефакт скачан в кэш: {downloaded_path}")
            
            # --- ИСПРАВЛЕНИЕ: Умный поиск config.json внутри структуры MLflow ---
            possible_hf_dirs = ["model", "components/model", "data/model", ""]
            
            for d in possible_hf_dirs:
                check_path = os.path.join(downloaded_path, d)
                if os.path.exists(os.path.join(check_path, "config.json")):
                    logger.info(f"Hugging Face веса найдены в подпапке: {check_path}")
                    self.loaded_from_mlflow = True
                    return check_path
            
            logger.warning("Не удалось найти config.json внутри скачанного артефакта MLflow.")
            self.loaded_from_mlflow = False
            return self.model_name_or_path
            
        except Exception as e:
            logger.warning(
                f"Не удалось загрузить модель из MLflow ({model_uri}). "
                f"Используем базовую ({self.model_name_or_path}). Ошибка: {e}"
            )
            self.loaded_from_mlflow = False
            return self.model_name_or_path

    def build(self) -> PreTrainedModel:
        module_name, class_name = self.auto_model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        load_path = self._resolve_load_path()
        logger.info(f"Загрузка модели из: {load_path}")

        bnb_config = None
        if self.quantization_config is not None:
            if isinstance(self.quantization_config, DictConfig):
                quant_dict = OmegaConf.to_container(self.quantization_config, resolve=True)
            else:
                quant_dict = self.quantization_config

            compute_dtype_str = quant_dict.get("bnb_4bit_compute_dtype")
            if isinstance(compute_dtype_str, str):
                quant_dict["bnb_4bit_compute_dtype"] = getattr(torch, compute_dtype_str)

            bnb_config = BitsAndBytesConfig(**quant_dict)

        parsed_dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"

        if bnb_config is not None:
            device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
        else:
            device_map = None

        model = model_class.from_pretrained(
            load_path,
            num_labels=self.num_labels,
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=parsed_dtype,
            device_map=device_map,
        )

        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
            if model.config.vocab_size != vocab_size:
                model.resize_token_embeddings(vocab_size)

        # Логика применения PEFT или ручной заморозки.
        # finetuning_type — это режим ПРОДОЛЖЕНИЯ обучения и применяется
        # всегда, независимо от того, откуда пришли стартовые веса (HF Hub
        # или MLflow Production): если finetuning_type=head_only, backbone
        # должен замораживаться и при дообучении поверх Production-модели.
        if self.finetuning_type == "peft" and self.peft_config is not None:
            logger.info(
                "Инициализация PEFT/LoRA адаптеров. Базовая модель будет заморожена автоматически."
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if bnb_config is not None:
                model = prepare_model_for_kbit_training(model)

            # --- ИСПРАВЛЕНИЕ: Проверяем тип объекта перед инициализацией ---
            if isinstance(self.peft_config, LoraConfig):
                # Гидра уже инициализировала объект
                lora_config = self.peft_config
            else:
                # Если это сырой словарь (например, без _target_)
                if isinstance(self.peft_config, DictConfig):
                    peft_dict = OmegaConf.to_container(self.peft_config, resolve=True)
                else:
                    peft_dict = dict(self.peft_config)
                
                # Удаляем _target_, чтобы kwargs не сломались
                peft_dict.pop("_target_", None)
                lora_config = LoraConfig(**peft_dict)

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        elif self.finetuning_type == "head_only":
            logger.info("Режим head_only: замораживаем backbone, оставляем обучаемым только классификатор.")
            for name, param in model.named_parameters():
                # Hugging Face обычно называет линейные слои классификатора 'classifier' или 'score'
                if "classifier" not in name and "score" not in name:
                    param.requires_grad = False

        elif self.finetuning_type == "full":
            logger.info("Режим full: обучаем все веса модели.")

        return model