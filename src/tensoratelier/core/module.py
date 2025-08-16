from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from torch.nn import Module
from torch.optim import Optimizer

from tensoratelier.core import AtelierOptimizer, AtelierTrainer

MODULE_OPTIMIZERS = Union[Optimizer, AtelierOptimizer]


class ModulePrereqs(ABC):
    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass


class AtelierModule(Module, ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._trainer: Optional[AtelierTrainer] = None
        self._metric_attributes: Optional[dict[int, str]] = None
        self._automatic_optimization: bool = True
        self._optimizer: Union[AtelierOptimizer, Optimizer]

    @abstractmethod
    def training_step(self, batch, batch_idx) -> Any:
        """
        Defines a single training step.

        Args:
            batch: The training batch data
            batch_idx: Index of the current batch

        Returns:
            Training loss or dict containing loss and other metrics
        """

    @abstractmethod
    def configure_optimizers(self) -> Union[AtelierOptimizer, Optimizer, dict]:
        """
        Configure and return the optimizer(s) for training.

        Returns:
            Optimizer instance or configuration dict
        """

    @property
    def trainer(self) -> AtelierTrainer:
        if self._trainer is None:
            raise RuntimeError(
                f"{self.__class__.__qualname__} is not attached to a {
                    AtelierTrainer.__name__
                }."
            )
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: AtelierTrainer):
        if isinstance(trainer, AtelierTrainer):
            self._trainer = trainer

    @property
    def optimizer(self) -> AtelierOptimizer:
        if self._optimizer is None:
            raise RuntimeError(
                f"{self.__class__.__qualname__} is not attached to a {
                    AtelierOptimizer.__name__
                }."
            )

        return self._optimizer

    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization) -> bool:
        self._automatic_optimization = automatic_optimization
