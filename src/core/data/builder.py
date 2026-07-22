# src/core/data/builder.py
"""
Модуль управления данных для NLP пайплайна.

Содержит реализацию PyTorch Lightning DataModule, который инкапсулирует
всю логику загрузки, очистки, кэширования и подготовки батчей текста.
Поддерживает динамическую токенизацию и версионирование кэша на основе
хэширования полной конфигурации обработки данных.
"""

import logging
import os
import json
import hashlib
from typing import Any, Optional
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate
from datasets import load_from_disk, DatasetDict, ClassLabel
from transformers import PreTrainedTokenizerBase

from src.core.data.datasets import NLPDatasetAdapter

logger = logging.getLogger(__name__)

class NLPDataModule(pl.LightningDataModule):
    """
    Lightning DataModule для обработки текстовых данных.
    """

    def __init__(self, data_cfg: Any, tokenizer: PreTrainedTokenizerBase):
        """
        Инициализирует DataModule и вычисляет путь для DVC кэша.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        
        hash_dict = {
            "cleaner": OmegaConf.to_container(self.data_cfg.cleaner, resolve=True),
            "source": OmegaConf.to_container(self.data_cfg.source, resolve=True),
            "text_column": self.data_cfg.get("text_column"),
            "target_column": self.data_cfg.get("target_column"),
            "seed": self.data_cfg.get("seed"),
            "max_length": self.data_cfg.get("max_length")
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True)
        config_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:8]
        
        dataset_name = self.data_cfg.get("dataset_name", "nlp_dataset")
        
        self.processed_dir = os.path.join(
            self.data_cfg.paths.processed_data_dir, 
            f"{dataset_name}_cleaned_{config_hash}"
        )


    def _compute_class_weights(self, labels: list[int]) -> torch.Tensor:
        """
        Вычисляет веса классов как 1 / frequency.
        Нормализует так чтобы среднее = 1.0 (стабильнее для lr).
        """
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights = weights / weights.mean()  # нормализация
        logger.info(
            f"Веса классов: " +
            ", ".join(f"класс {i}: {w:.3f}" for i, w in enumerate(weights))
        )
        return torch.tensor(weights, dtype=torch.float)
    

    def _make_sampler(
        self, labels: list[int], replacement: bool
    ) -> WeightedRandomSampler:
        """
        WeightedRandomSampler — каждый пример тянется с вероятностью
        обратно пропорциональной частоте его класса.
        replacement=False → undersampling (мажоритарный класс режется)
        replacement=True  → oversampling (миноритарный класс дублируется)
        """
        counts = np.bincount(labels)
        class_weights = 1.0 / counts
        sample_weights = torch.tensor(
            [class_weights[label] for label in labels],
            dtype=torch.float
        )

        # replacement=False: num_samples = 2 * min_class_count
        # replacement=True:  num_samples = len(dataset) как обычно
        if not replacement:
            num_samples = int(counts.min() * len(counts))
            logger.info(
                f"Undersampling через WeightedRandomSampler: "
                f"{num_samples} примеров на эпоху "
                f"(исходно {len(labels)})"
            )
        else:
            num_samples = len(labels)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=replacement,
        )


    def _apply_undersampling(
        self, dataset, labels: list[int], target_ratio: float
    ):
        """
        Явный undersampling: случайно выбрасывает примеры мажоритарного
        класса до target_ratio. target_ratio=2.0 → класс0 : класс1 = 2:1
        """
        counts = np.bincount(labels)
        min_class = counts.argmin()
        min_count = counts[min_class]
        labels_array = np.array(labels)

        keep_indices = []
        for cls in range(len(counts)):
            cls_indices = np.where(labels_array == cls)[0]
            if cls == min_class:
                keep_indices.extend(cls_indices.tolist())
            else:
                # Сколько оставить мажоритарного
                keep_count = int(min_count * target_ratio)
                kept = np.random.choice(
                    cls_indices, size=keep_count, replace=False
                )
                keep_indices.extend(kept.tolist())

        np.random.shuffle(keep_indices)
        logger.info(
            f"Undersampling: {len(labels)} → {len(keep_indices)} примеров. "
            f"Новый баланс: " +
            ", ".join(
                f"класс {i}: {np.sum(labels_array[keep_indices] == i)}"
                for i in range(len(counts))
            )
        )
        return dataset.select(keep_indices)
    

    def prepare_data(self) -> None:
        """
        Скачивает сырые данные, применяет очистку и сохраняет результат на диск.
        """
        if os.path.exists(self.processed_dir) and not self.data_cfg.get("force_reprocess", False):
            logger.info(f"Нашли кэш данных: {self.processed_dir}. Очистка пропущена.")
            return

        logger.info("Начинаем загрузку и обработку сырых данных...")
        
        raw_datasets = instantiate(self.data_cfg.source).load()
        
        # Разделение на train/val/test
        if "validation" in raw_datasets and "test" in raw_datasets:
            raw_train = raw_datasets["train"]
            raw_val = raw_datasets["validation"]
            raw_test = raw_datasets["test"]
        else:
            raw_datasets["train"] = raw_datasets["train"].cast_column(
                "label", ClassLabel(names=[0, 1])
            )
            stratify_col = self.data_cfg.get("stratify_column", None)
            # Fallback: Если сплитов нет, делаем их искусственно из train
            split_ds = raw_datasets["train"].train_test_split(
                test_size=self.data_cfg.val_split_size * 2, # Берем x2, чтобы поделить на val и test
                seed=self.data_cfg.seed,
                stratify_by_column=stratify_col
            )
            raw_train = split_ds["train"]
            
            # Делим отщипнутый кусок пополам между val и test
            val_test_split = split_ds["test"].train_test_split(
                test_size=0.5,
                seed=self.data_cfg.seed,
                stratify_by_column=stratify_col
            )
            raw_val = val_test_split["train"]
            raw_test = val_test_split["test"]

        cleaner_pipeline = instantiate(self.data_cfg.cleaner)

        # Функция для минимизации дублирования кода
        def _process_split(dataset_split):
            return NLPDatasetAdapter(
                hf_dataset=dataset_split, 
                text_column=self.data_cfg.text_column, 
                cleaning_pipeline=cleaner_pipeline,
                num_proc=self.data_cfg.get("preprocessing_num_workers", 4),
                batch_size=self.data_cfg.get("preprocessing_batch_size", 1000)
            ).prepare_dataset()

        processed_dataset = DatasetDict({
            "train": _process_split(raw_train),
            "validation": _process_split(raw_val),
            "test": _process_split(raw_test)
        })
        
        processed_dataset.save_to_disk(self.processed_dir)
        logger.info(f"Данные успешно очищены и сохранены в {self.processed_dir}")

    def setup(self, stage=None):
        # --- ПРЕДОХРАНИТЕЛЬ ОТ ДВОЙНОГО ВЫЗОВА ---
        if stage in ("fit", None) and getattr(self, "train_dataset", None) is not None:
            logger.info("DataModule.setup уже инициализирован, пропускаем повторный вызов.")
            return

        processed_dataset = load_from_disk(self.processed_dir)
        balancing_cfg = self.data_cfg.get("balancing", {})

        if stage in ("fit", None):
            train_ds = processed_dataset["train"]
            labels = train_ds[self.data_cfg.target_column]

            undersampling_cfg = balancing_cfg.get("undersampling", {})
            if undersampling_cfg and undersampling_cfg.get("enabled", False):
                train_ds = self._apply_undersampling(
                    train_ds, labels,
                    undersampling_cfg.get("target_ratio", 2.0),
                )
                labels = train_ds[self.data_cfg.target_column]

            self.train_dataset = train_ds

            sampler_cfg = balancing_cfg.get("sampler", {})
            if sampler_cfg and sampler_cfg.get("enabled", False):
                self.train_sampler = self._make_sampler(
                    labels, replacement=sampler_cfg.get("replacement", False)
                )
            else:
                self.train_sampler = None

            class_weights_cfg = balancing_cfg.get("class_weights")
            if class_weights_cfg == "auto":
                self.class_weights = self._compute_class_weights(labels)
            elif isinstance(class_weights_cfg, list):
                self.class_weights = torch.tensor(class_weights_cfg, dtype=torch.float)
            else:
                self.class_weights = None

            self.val_dataset = processed_dataset["validation"]

        if stage == "validate":
            self.val_dataset = processed_dataset["validation"]

        if stage in ("test", None):
            self.test_dataset = processed_dataset["test"]

        self.collator = instantiate(
            self.data_cfg.collator, tokenizer=self.tokenizer
        )
    def train_dataloader(self):
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.train_dataset,
            collate_fn=self.collator,
            shuffle=self.train_sampler is None,  # shuffle только если нет sampler
            sampler=self.train_sampler,
        )

    def val_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.val_dataset,
            collate_fn=self.collator,
            shuffle=False
        )
        
    def test_dataloader(self) -> Any:
        return instantiate(
            self.data_cfg.dataloader,
            dataset=self.test_dataset,
            collate_fn=self.collator,
            shuffle=False
        )