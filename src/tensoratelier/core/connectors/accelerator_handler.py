from typing import Type, Union, TYPE_CHECKING

import torch

from tensoratelier.accelerators import BaseAccelerator
from tensoratelier.accelerators.registry import ACCELERATOR_REGISTRY

if TYPE_CHECKING:
    from tensoratelier.core import AtelierModule


class AcceleratorHandler:
    def __init__(self, accelerator: Union[str, torch.device] = "auto"):
        if self._accelerator_flag == "auto":
            self._accelerator: Type[BaseAccelerator] = self._auto_select_accelerator(
            )
        else:
            self._accelerator: Type[BaseAccelerator] = self._check_accelerator_flag(
                accelerator=accelerator
            )

        # self._accelerator is the uninstatiated classobject.

    def _move_batch(self, batch):
        return batch.to(self._device)

    def _move_model(self, model):
        model.to(self._accelerator_flag)

    def _check_accelerator_flag(accelerator: Union[str, torch.device]):
        accelerator_cls = ACCELERATOR_REGISTRY._get(accelerator)

        if not accelerator_cls:
            raise ValueError(
                f"You selected an invalid accelerator name: 'accelerator={
                    accelerator!r}'. Available names are: auto, {', '.join(ACCELERATOR_REGISTRY)}."
            )

    def _auto_select_accelerator():
        if torch.cuda.is_available() and "cuda" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("cuda")
        elif torch.backends.mps.is_available() and "mps" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("mps")
        elif torch.xpu.is_available() and "xpu" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("xpu")
        else:
            return "cpu"
