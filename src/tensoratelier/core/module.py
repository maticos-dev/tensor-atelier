from typing import Any, Optional, Union

from torch.nn import Module
from torch.optim import Optimizer

from tensoratelier.core import AtelierOptimizer, AtelierTrainer

MODULE_OPTIMIZERS = Union[Optimizer, AtelierOptimizer]


class AtelierModule(Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._trainer: Optional[AtelierTrainer] = None
        self._metric_attributes: Optional[dict[int, str]] = None

    @property
    def trainer(self) -> AtelierTrainer:
        if self._trainer is None:
            raise RuntimeError(
                f"{self.__class__.__qualname__} is not attached to a {
                    AtelierTrainer.__name__
                }."
            )
