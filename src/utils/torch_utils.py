# src/utils/torch_utils.py

import functools
import logging

import torch


logger = logging.getLogger(__name__)


def register_safe_globals() -> None:
    safe_classes = [
        functools.partial,
        torch.optim.AdamW,
    ]

    # OmegaConf — все типы которые могут попасть в чекпоинт через save_hyperparameters
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import (
            AnyNode,
            BooleanNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            StringNode,
        )

        safe_classes.extend(
            [
                ListConfig,
                DictConfig,
                ContainerMetadata,
                Metadata,
                AnyNode,
                IntegerNode,
                FloatNode,
                BooleanNode,
                StringNode,
                EnumNode,
            ]
        )
    except ImportError:
        pass

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
        f"Зарегистрированы safe globals: "
        f"{[c.__name__ if hasattr(c, '__name__') else str(c) for c in safe_classes]}"
    )
