from typing import Type, Union

import torch

from tensoratelier.accelerators import BaseAccelerator


class _AcceleratorRegistry:
    def __init__(self):
        self._registry = {}

    def _register(self, accelerator_type: str):
        def wrapper(cls: Type[BaseAccelerator]):
            if accelerator_type in self._registry:
                raise ValueError("Accelerator 'name' already registered.")

            self._registry[accelerator_type] = cls

        return wrapper

    def _get(self, accelerator_type: Union[str, torch.device]) -> BaseAccelerator:
        key = (
            accelerator_type.type.lower()
            if isinstance(accelerator_type, torch.device)
            else accelerator_type.lower()
        )

        try:
            return self._registry[key]()
        except KeyError:
            raise ValueError(
                f"Unsupported accelerator '{
                    key
                }'. Only 'cpu' is supported at this time."
            )


ACCELERATOR_REGISTRY = _AcceleratorRegistry()
