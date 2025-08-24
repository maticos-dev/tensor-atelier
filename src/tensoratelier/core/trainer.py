from __future__ import annotations
import logging
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from tensoratelier.handlers import AcceleratorHandler
from tensoratelier.handlers.utils import auto_move_dataloader, auto_move_model
from tensoratelier.loops import _FitLoop, _TrainingEpochLoop
from tensoratelier.profilers import (
    BaseProfiler,
    _FittingProfiler,
    _OptimizationProfiler,
)
from tensoratelier.utils.parsing import _wrap_args

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensoratelier.core import AtelierDataLoader, AtelierModule, AtelierOptimizer


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

    @_wrap_args
    @auto_move_dataloader
    @auto_move_model
    def fit(
        self,
        module: AtelierModule,
        dataloader: Optional[AtelierDataLoader] = None,
        train_val_split: Optional[Union[torch.Tensor,
                                        np.ndarray, list, tuple]] = None,
        max_epochs: int = 0,
    ) -> None:

        self.module = module

        if isinstance(dataloader, AtelierDataLoader):
            self.attach_trainer_to_dataloader(dataloader)
            self.dataloader = dataloader

        optimizer = self.module.configure_optimizers()
        self.validate_optimizer(optimizer)

        self._train_profiler = _FittingProfiler()
        self._optim_profiler = _OptimizationProfiler()

        # initialize and run fit loop.
        self.init_fit_loop(dataloader, max_epochs)
        self.fit_loop.run()

    def attach_trainer_to_dataloader(self, dataloader: AtelierDataLoader):
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

        self.fit_loop = _FitLoop(
            self, self.atelier_module, train_loader, max_epochs)
        self.fit_loop.epoch_loop = _TrainingEpochLoop(self, max_epochs)

    def validate_optimizer(self, optimizer) -> bool:
        return (isinstance(optimizer, torch.optim.Optimizer) or
                isinstance(optimizer, AtelierOptimizer))

    def training_step(self) -> Tensor:  # called by automatic optimization.
        with self.train_profiler.profile("Executing module training step",
                                         self.fit_loop._epoch_progress.epoch_idx):
            batch_idx = self.fit_loop.epoch_loop._batch_progress.batch_idx
            batch = self.dataloader.forward()
            return self.atelier_module.training_step(batch, batch_idx)

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

    def zero_grad_step(self, optimizer):
        # if atelier_module linked and implements this,
        # use user implementation.
        # otherwise, do smth.
        # maybe like a learning rate, which would output
        # according to batch states saved in training_epoch_loop
        # state
        pass

    @property
    def epoch_progress(self):
        return self.fit_loop._epoch_progress

    @property
    def atelier_module(self):
        if hasattr(self, "module") and self.module is not None:
            return self.module
        else:
            raise AttributeError(
                f"No module associated with trainer instance {
                    self.__class__.__qualname__
                }."
            )

    @property
    def atelier_optimizer(self):
        # self.lightning_module.optimizers
        return self.atelier_module.optimizer

    @property
    def train_profiler(self) -> BaseProfiler:
        if getattr(self, "_profiler", False):
            raise AttributeError(
                "No training profiler linked to trainer object."
            )
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
    def optimization_profiler(self, profiler_obj: BaseProfiler
                              ) -> Optional[BaseProfiler]:
        if not isinstance(profiler_obj, BaseProfiler):
            raise TypeError(
                f"Expected profiler instance to be of \
            type BaseProfiler, but got {profiler_obj.__class__.__qualname__}"
            )

        log.debug(f"Attached {profiler_obj.__class__.__qualname__} to trainer")

        self._optim_profiler = profiler_obj
