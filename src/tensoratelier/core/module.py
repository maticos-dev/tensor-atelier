from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, TYPE_CHECKING

from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from tensoratelier.mixins import AttributeOverrideMixin

if TYPE_CHECKING:
    from tensoratelier.core import AtelierOptimizer, AtelierTrainer


class ModulePrereqs(ABC):
    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass


class AtelierModule(AttributeOverrideMixin, Module, ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        super(AtelierModule, self).__init__(*args, **kwargs)
        self._trainer: Optional[AtelierTrainer] = None
        self._metric_attributes: Optional[dict[int, str]] = None
        self._automatic_optimization: bool = True
        self._optimizer: Optional[Optimizer] = None

    @abstractmethod
    def training_step(self, batch, batch_idx) -> Tensor:
        """Defines a single training step.

        Args:
            batch: The training batch data
            batch_idx: Index of the current batch

        Returns:
            Training loss or dict containing loss and other metrics
        """

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        """Configure and return the optimizer(s) for training.

        Returns:
            Optimizer instance or configuration dict
        """

    @property
    def trainer(self) -> AtelierTrainer:
        if self._trainer is None:
            raise RuntimeError(
                f"{self.__class__.__qualname__} is not attached to a {AtelierTrainer.__name__}."
            )
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: AtelierTrainer):
        if isinstance(trainer, AtelierTrainer):
            self._trainer = trainer

    @property
    def optimizer(self) -> Optimizer:
        if self._optimizer is None:
            raise RuntimeError(
                f"{self.__class__.__qualname__} is not attached to an optimizer.")

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        if isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            raise TypeError(f"Module optimizer must be of type {Optimizer.__qualname__}")


    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization):
        self._automatic_optimization = automatic_optimization
