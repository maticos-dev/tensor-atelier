from typing import Type, Union

import torch

from tensoratelier.accelerators import BaseAccelerator


class _AcceleratorRegistry:
    def __init__(self):
        self._registry: dict[str, dict[str, type]] = {}

    def _register(self, accelerator_name: str, variant: str = "default"):
        assert accelerator_name in ("gpu", "tpu", "cpu")

        def wrapper(cls: Type[BaseAccelerator]):
            if accelerator_name.lower() in self._registry:
                raise ValueError("Accelerator 'name' already registered.")
            self._registry[accelerator_name] = {}
            self._registry[accelerator_name][variant] = cls

        return wrapper

    def _get(
        self, accelerator_name: Union[str, torch.device], variant: str = "default"
    ) -> BaseAccelerator:
        name = (
            accelerator_name.type.lower()
            if isinstance(accelerator_name, torch.device)
            else accelerator_name.lower()
        )

        try:
            if variant not in self._registry[name]:
                raise ValueError(
                    f"Unsupported variant '{
                        variant}' for accelerator '{name}'."
                )
            return self._registry[name][variant]()
        except KeyError:
            raise ValueError(
                f"Unsupported accelerator '{
                    name}'. Only 'cpu' is supported at this time."
            )


ACCELERATOR_REGISTRY = _AcceleratorRegistry()
