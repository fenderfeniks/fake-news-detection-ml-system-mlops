import logging
import os

from datasets import load_dataset, load_from_disk
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)


class RawDataFetcher:
    """
    Универсальный класс для получения сырых данных.
    Проверяет локальное наличие, при необходимости скачивает с Kaggle или HuggingFace.
    """

    def __init__(
        self,
        source_type: str,
        raw_dir: str,
        dataset_name: str = None,
        file_name: str = None,
        # ИЗМЕНЕНО: token убран — Kaggle читает KAGGLE_USERNAME / KAGGLE_KEY из env сам.
        # Параметр оставлен только для HuggingFace (HF_TOKEN).
        token: str = None,
        **kwargs,
    ):
        self.source_type = source_type
        self.raw_dir = raw_dir
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.token = token  # используется только в _load_hf
        self.kwargs = kwargs

    def load(self):
        """Единая точка входа для получения DatasetDict."""
        os.makedirs(self.raw_dir, exist_ok=True)

        if self.source_type == "local_csv":
            return self._load_local()
        elif self.source_type == "kaggle":
            return self._load_kaggle()
        elif self.source_type == "hf":
            return self._load_hf()
        else:
            raise ValueError(f"Неизвестный тип источника данных: {self.source_type}")

    def _load_local(self):
        file_path = os.path.join(self.raw_dir, self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Критическая ошибка: Локальный файл {file_path} не найден! "
                "Данные должны лежать в этой папке."
            )
        logger.info(f"Загрузка локального файла: {file_path}")
        return load_dataset("csv", data_files=file_path, **self.kwargs)

    def _load_kaggle(self):
        file_path = os.path.join(self.raw_dir, self.file_name)

        if os.path.exists(file_path):
            logger.info(f"Kaggle датасет найден локально: {file_path}. Скачивание пропущено.")
            return load_dataset("csv", data_files=file_path, **self.kwargs)

        # ИЗМЕНЕНО: убрана логика token.split(':').
        # kaggle-библиотека сама читает KAGGLE_USERNAME и KAGGLE_KEY из env при authenticate().
        # Они прокидываются через K8s Secret → KubernetesPodOperator env_vars.
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            raise EnvironmentError(
                "KAGGLE_USERNAME и KAGGLE_KEY не найдены в env. "
                "Проверь K8s Secret 'fake-news-api-secrets' и прокидывание "
                "через KubernetesPodOperator в retrain_model_dag."
            )

        logger.info(f"Скачиваем {self.dataset_name} с Kaggle (user: {username})...")

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_name, path=self.raw_dir, unzip=True)
        logger.info("Скачивание с Kaggle завершено.")

        return load_dataset("csv", data_files=file_path, **self.kwargs)

    def _load_hf(self):
        hf_local_path = os.path.join(self.raw_dir, self.dataset_name.replace("/", "_"))

        if not os.path.exists(hf_local_path):
            logger.info(
                f"Данные не найдены локально. Скачиваем {self.dataset_name} из HuggingFace..."
            )
            # token здесь — HF_TOKEN для приватных репо, для публичных = None
            dataset = load_dataset(self.dataset_name, token=self.token, **self.kwargs)
            dataset.save_to_disk(hf_local_path)
            logger.info(f"HF датасет сохранен в {hf_local_path}")
            return dataset

        logger.info(f"HF датасет найден локально: {hf_local_path}")
        return load_from_disk(hf_local_path)