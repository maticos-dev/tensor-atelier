import logging
from typing import Union, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from tensoratelier.core.connectors import AcceleratorHandler
from tensoratelier.core.connectors.utils import auto_move_dataloader, auto_move_model
from tensoratelier.loops import _FitLoop, _TrainingEpochLoop
from tensoratelier.profilers import (
    _FittingProfiler,
    _OptimizationProfiler,
)
from tensoratelier.utils import _wrap_args

if TYPE_CHECKING:
    from tensoratelier.core import AtelierDataLoader, AtelierModule
    from tensoratelier.loops.progress import _EpochProgress
    from tensoratelier.profilers import BaseProfiler

log = logging.getLogger(__name__)


class AtelierTrainer:
    @_wrap_args
    def __init__(
        self,
        *,
        max_epochs: int = 10,
        accelerator: Union[str, torch.device],
        profiler: BaseProfiler,
    ) -> None:
        self.max_epochs = max_epochs
        self._accelerator_handler = AcceleratorHandler(accelerator)
        self._profiler = _FittingProfiler()  # TODO MAKE THIS PROPERTY

    @auto_move_dataloader
    @auto_move_model
    def fit(
        self,
        model: AtelierModule,
        dataloader: AtelierDataLoader,
        train_val_split: Union[torch.Tensor, np.ndarray, list, tuple],
        max_epochs: int = 0,
    ) -> None:
        self.init_fit_loop(model, dataloader, max_epochs)
        self.attach_trainer_to_dataloaders(dataloader)

        self._train_profiler = _FittingProfiler()
        self._optim_profiler = _OptimizationProfiler()

        optimizer = model.configure_optimizers()
        self.validate_optimizer(optimizer)

        self.fit_loop.run()

    def attach_trainer_to_dataloaders(self, dataloader: AtelierDataLoader):
        if isinstance(dataloader, AtelierDataLoader):
            log.debug(
                f"Attached {self.__class__.__qualname__} to {
                    dataloader.__class__.__qualname__
                }"
            )
            dataloader.trainer = self

    def init_fit_loop(self, train_loader, max_epochs):
        log.debug(
            f"Initialized fit and nested epoch loops for {
                self.__class__.__qualname__}"
        )

        self.fit_loop = _FitLoop(self, train_loader, max_epochs)
        self.fit_loop.epoch_loop = _TrainingEpochLoop(self, max_epochs)

    def training_step(self):
        with self.profiler.profile("Module training step"):
            self.atelier_module.training_step(batch, batch_idx)

    def optimizer_step(self, loss: Tensor):
        if not isinstance(self.atelier_optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected type 'torch.optim.optimizer' but got {
                    self.atelier_optimizer.__class__.__name__
                }"
            )

        self.atelier_optimizer.step()
        self.atelier_optimizer.zero_grad()
        loss.backward()

    @property
    def epoch_progress(self) -> _EpochProgress:
        return self.fit_loop._epoch_progress

    @property
    def atelier_module(self):
        if hasattr(self, "model") and self.model is not None:
            return self.model
        else:
            raise AttributeError(
                f"No model associated with trainer instance {
                    self.__class__.__qualname__
                }."
            )

    @property
    def atelier_optimizer(self) -> list:
        # self.lightning_module.optimizers
        return self.atelier_module.optimizer

    @property
    def train_profiler(self) -> BaseProfiler:
        if getattr(self, "_profiler", False):
            raise AttributeError(
                "No training profiler linked to trainer object.")
        return self._train_profiler

    @train_profiler.setter
    def train_profiler(self, profiler_obj: BaseProfiler) -> None:
        if not isinstance(profiler_obj, BaseProfiler):
            raise TypeError(
                f"Expected profiler instance to be of \
            type BaseProfiler, but got {profiler_obj.__class__.__qualname__}"
            )

        log.debug(f"Attached {profiler_obj.__class__.__qualname__} to trainer")

        self._train_profiler = profiler_obj

    @property
    def optimization_profiler(self) -> BaseProfiler:
        if getattr(self, "_profiler", False):
            raise AttributeError(
                "No optimization profiler linked to trainer object.")
        return self._optim_profiler

    @optimization_profiler.setter
    def optimization_profiler(self, profiler_obj: BaseProfiler) -> BaseProfiler:
        if not isinstance(profiler_obj, BaseProfiler):
            raise TypeError(
                f"Expected profiler instance to be of \
            type BaseProfiler, but got {profiler_obj.__class__.__qualname__}"
            )

        log.debug(f"Attached {profiler_obj.__class__.__qualname__} to trainer")

        self._optim_profiler = profiler_obj
