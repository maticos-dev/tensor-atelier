from torch.nn import Module
from typing import Any, Callable, Optional, Union
from torch.optim import Optimizer

from tensoratelier.core import AtelierOptimizer, AtelierTrainer

MODULE_OPTIMIZERS = Union[Optimizer, AtelierOptimizer]

class AtelierModule(Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._trainer: Optional[AtelierTrainer] = None
        self._metric_attributes: Optional[dict[int, str]] = None

    def optimizers(self, use_atelier_optimizer: bool = True):

        if use_atelier_optimizer:
            opts = self.trainer.strategy._lightning_optimizers
        else:
            opts = self.trainer.optimizers

        # single optimizer
        if (
            isinstance(opts, list)
            and len(opts) == 1
            and isinstance(opts[0], MODULE_OPTIMIZERS)
        )

    @property
    def trainer(self) -> AtelierTrainer:
        if self._trainer is None:
            raise RuntimeError(f"{self.__class__.__qualname__} is not attached to a {AtelierTrainer.__name__}.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Optional[AtelierTrainer]) -> None:
        self._trainer = trainer