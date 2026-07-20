import logging
import os
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def init_nlp_notebook(config_name: str = "main"):
    """
    Инициализирует окружение для NLP ноутбука.
    """
    # 1. Находим корень проекта: ищем папку, где лежит 'pyproject.toml' или 'src'
    # Берем текущую рабочую директорию (обычно это корень проекта при запуске VS Code)
    project_root = Path.cwd()

    # Если мы глубоко в ноутбуках, поднимаемся на один уровень вверх (на всякий случай)
    if project_root.name == "notebooks":
        project_root = project_root.parent

    # 2. Добавляем корень в sys.path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # 3. Инициализация Hydra
    GlobalHydra.instance().clear()

    # Важно: config_path должен быть относительным от ЭТОГО файла
    # Если этот файл в src/utils, то configs лежат в ../../configs
    try:
        initialize(config_path="../../configs", version_base="1.3")
    except ValueError:
        # Если инициализация уже была, очищаем и пробуем снова
        GlobalHydra.instance().clear()
        initialize(config_path="../../configs", version_base="1.3")

    cfg = compose(config_name=config_name)

    # 4. Настройка логирования
    logging.basicConfig(level=logging.INFO, force=True)

    print(f"NLP Environment ready. Root: {project_root}")
    return cfg
