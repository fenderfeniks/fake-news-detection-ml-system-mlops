# src/utils/hydra_utils.py
import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.config_schema import ConfigSchema


# Опционально импортируем python-json-logger для продакшена
try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

logger = logging.getLogger(__name__)


def _force_utf8_console_encoding() -> None:
    """
    На Windows консоль по умолчанию использует системную локаль (часто
    cp1251), а не UTF-8. Логи с эмодзи или другими символами вне cp1251
    (например , ) роняют logging с UnicodeEncodeError — не критично
    для выполнения скрипта (logging сам ловит ошибку), но замусоривает
    вывод трейсбэками и теряет часть сообщений.

    Переключаем существующие StreamHandler'ы (включая handlers на
    root-логгере, которые ставит Hydra по умолчанию) на UTF-8 через
    stream.reconfigure — доступно в Python 3.7+, no-op на не-Windows,
    где stdout/stderr обычно уже UTF-8.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="backslashreplace")
                except (ValueError, OSError):
                    # Поток уже закрыт/не поддерживает reconfigure — пропускаем.
                    pass


def setup_config(cfg: DictConfig) -> DictConfig:
    """
    Валидирует конфиг, разрешает ссылки и выводит его в логи.
    Возвращает валидированный конфиг.
    """

    _force_utf8_console_encoding()

    # Автоматический перевод логов в JSON формат (для ELK/Loki)
    # Срабатывает только если установлен пакет python-json-logger и мы в PROD
    if HAS_JSON_LOGGER and os.getenv("ENVIRONMENT") == "prod":
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            handler.setFormatter(formatter)

    # Открываем конфиг для редактирования (на случай если он заблокирован)
    OmegaConf.set_struct(cfg, False)

    # Динамически вычисляем абсолютный путь к корню проекта.
    # Если этот файл лежит в src/core/utils/hydra_utils.py, то нужно подняться на 4 уровня вверх.
    # Если в src/utils/hydra_utils.py, то на 3 уровня (.parents[2]).
    # В данном случае поднимаемся из src/core/utils в корень (индекс 3):
    project_root = str(Path(__file__).resolve().parents[2])

    # Жестко прописываем абсолютный путь
    cfg.paths.root_dir = project_root

    # 1. Разрешаем все ссылки вида ${paths.log_dir}
    OmegaConf.resolve(cfg)

    # 2. Строгая валидация (сверяем YAML с нашим датаклассом Schema)
    schema = OmegaConf.structured(ConfigSchema)

    # Сохраняем результат слияния схемы и пользовательского конфига
    validated_cfg = OmegaConf.merge(schema, cfg)

    # Блокируем конфиг от случайных изменений в дальнейшем коде
    OmegaConf.set_struct(validated_cfg, True)

    # 3. Заменили info на debug, чтобы конфиг не засорял терминал, но оставался в файловых логах
    logger.debug(f"Финальная конфигурация эксперимента:\n{OmegaConf.to_yaml(validated_cfg)}")

    return validated_cfg
