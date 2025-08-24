from typing import Type, Union, TYPE_CHECKING

import torch

from tensoratelier.accelerators import BaseAccelerator
from tensoratelier.accelerators.registry import ACCELERATOR_REGISTRY
from tensoratelier.utils.parsing import _wrap_args
from tensoratelier.handlers.utils import check_accelerator_flag

if TYPE_CHECKING:
    from tensoratelier.core import AtelierModule


class AcceleratorHandler:
    @_wrap_args
    @check_accelerator_flag
    def __init__(self, accelerator_flag: Union[str, torch.device] = "auto"):
        self._accelerator_flag = accelerator_flag

        if self._accelerator_flag == "auto":
            self._accelerator: Type[BaseAccelerator] = self._auto_select_accelerator(
            )

        else:
            self._accelerator: Type[BaseAccelerator] = self._accelerator_flag

        # self._accelerator is the uninstatiated classobject.

    def _move_batch(self, batch):
        return batch.to(self._accelerator)

    def _move_model(self, model):
        # need to move to torch.device,
        #
        model.to(self._accelerator)

    def _auto_select_accelerator(self):
        if torch.cuda.is_available() and "cuda" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("cuda")
        elif torch.backends.mps.is_available() and "mps" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("mps")
        elif torch.xpu.is_available() and "xpu" in ACCELERATOR_REGISTRY:
            return ACCELERATOR_REGISTRY._get("xpu")
        else:
            return "cpu"
