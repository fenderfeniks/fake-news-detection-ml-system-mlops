"""
Формирование pip_requirements для артефактов MLflow из pyproject.toml.

MLflow при mlflow.transformers.log_model(...) по умолчанию пытается сам
определить нужные зависимости (get_default_pip_requirements) — и в этот
список может попасть пакет, которого нет в окружении (например,
torchvision), из-за чего log_model падает с ModuleNotFoundError.

Вместо того чтобы хардкодить список пакетов в коде, читаем его из
единственного источника правды — pyproject.toml, группа
[project.optional-dependencies.inference-core]. Так requirements
артефакта модели всегда соответствуют тому, что реально объявлено
как нужное для инференса, без дублирования и рассинхронизации.
"""

import logging
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


logger = logging.getLogger(__name__)

_INFERENCE_GROUP = "inference-core"


def _strip_version_specifier(requirement: str) -> str:
    """'torch>=2.0.0' -> 'torch'. Также отбрасывает extras вида 'uvicorn[standard]'."""
    name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
    return name


def get_inference_pip_requirements(pyproject_path: str | Path) -> list[str]:
    """
    Читает [project.optional-dependencies.inference-core] из pyproject.toml
    и возвращает список requirements с версиями, реально установленными
    в текущем окружении (а не теми, что указаны в pyproject как диапазон).

    Это гарантирует, что MLflow-артефакт задекларирует именно ту версию,
    с которой модель была обучена и сохранена, а не абстрактный ">=".
    """
    pyproject_path = Path(pyproject_path)
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        declared = data["project"]["optional-dependencies"][_INFERENCE_GROUP]
    except KeyError:
        logger.warning(
            f"Группа [project.optional-dependencies.{_INFERENCE_GROUP}] "
            f"не найдена в {pyproject_path}. Возвращаю пустой список pip_requirements — "
            f"MLflow будет пытаться определить зависимости автоматически."
        )
        return []

    pinned: list[str] = []
    for requirement in declared:
        pkg_name = _strip_version_specifier(requirement)
        try:
            installed_version = version(pkg_name)
            pinned.append(f"{pkg_name}=={installed_version}")
        except PackageNotFoundError:
            logger.warning(
                f"Пакет '{pkg_name}' объявлен в группе '{_INFERENCE_GROUP}', "
                f"но не установлен в текущем окружении — пропускаю."
            )

    return pinned
