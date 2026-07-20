"""
Утилиты для безопасной загрузки PyTorch/Lightning чекпоинтов.

Начиная с PyTorch 2.6 дефолтное значение `weights_only` в `torch.load`
изменено на `True`. Чекпоинты PyTorch Lightning (.ckpt) содержат не
только веса модели, но и состояние оптимизатора, LR-шедулера и т.д. —
эти объекты не входят в белый список безопасных глобалов по умолчанию,
поэтому загрузка падает с `UnpicklingError`.

Эта утилита регистрирует классы, которые реально встречаются в наших
чекпоинтах, как доверенные глобалы. Используется в train.py и eval.py
перед любой загрузкой .ckpt файлов.
"""

import functools
import logging

import torch


logger = logging.getLogger(__name__)


def register_safe_globals() -> None:
    """
    Регистрирует классы, необходимые для загрузки Lightning-чекпоинтов
    через `torch.load(..., weights_only=True)`.

    Вызывать один раз до первого `torch.load` / `load_from_checkpoint`
    в процессе (train.py, eval.py).
    """
    safe_classes = [
        functools.partial,
        torch.optim.AdamW,
    ]

    try:
        import torch.optim.lr_scheduler as lr_scheduler

        safe_classes.extend(
            [
                lr_scheduler.CosineAnnealingLR,
                lr_scheduler.CosineAnnealingWarmRestarts,
                lr_scheduler.LambdaLR,
                lr_scheduler.OneCycleLR,
                lr_scheduler.ReduceLROnPlateau,
                lr_scheduler.StepLR,
            ]
        )
    except ImportError:
        pass

    torch.serialization.add_safe_globals(safe_classes)
    logger.debug(
        f"Зарегистрированы safe globals для torch.load: "
        f"{[c.__name__ if hasattr(c, '__name__') else str(c) for c in safe_classes]}"
    )
