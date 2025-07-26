from typing import Dict, Type, Union

import torch

from tensoratelier.accelerators import BaseAccelerator

_ACCELERATOR_REGISTRY: Dict[str, Type[BaseAccelerator]] = {}


def _register_accelerator(accelerator_type: str):
    def wrapper(cls: Type[BaseAccelerator]):
        if accelerator_type in _ACCELERATOR_REGISTRY:
            raise ValueError("Accelerator 'name' already registered.")

        _ACCELERATOR_REGISTRY[accelerator_type] = cls

    return wrapper


def _get_accelerator(accelerator_type: Union[str, torch.device]) -> BaseAccelerator:
    key = (
        accelerator_type.type.lower()
        if isinstance(accelerator_type, torch.device)
        else accelerator_type.lower()
    )

    try:
        return _ACCELERATOR_REGISTRY[key]()
    except KeyError:
        raise ValueError(
            f"Unsupported accelerator '{
                key}'. Only 'cpu' is supported at this time."
        )
