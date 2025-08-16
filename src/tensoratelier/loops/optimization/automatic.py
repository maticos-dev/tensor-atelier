from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import Tensor

from tensoratelier.loops.progress import _OptimizationProgress

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensoratelier.core import AtelierTrainer


@dataclass
class _ClosureResult:
    loss: Optional[Tensor] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.loss is not None:
            self.loss = self.loss.detach().clone()


class Closure:
    def __init__(
        self,
        step_fn: Callable[[], _ClosureResult],
        backward_fn: Optional[Callable[[Tensor], None]] = None,
        zero_grad_fn: Optional[Callable[[], None]] = None,
    ):
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn

    @torch.enable_grad()
    def closure(self) -> _ClosureResult:
        step_output = self._step_fn()

        if step_output.loss is None:
            log.warning("'training_step' returned 'None'")

        if self._zero_grad_fn is not None:
            self._zero_grad_fn()

        if self._backward_fn is not None and step_output.closure_loss is not None:
            self._backward_fn(step_output.loss)

        return step_output

    def __call__(self):
        self._result = self.closure()
        return self._result.loss


class _AutomaticOptimization:
    def __init__(self, trainer: AtelierTrainer, optimizer) -> None:
        self.optim_progress: _OptimizationProgress = _OptimizationProgress()
        self.optimizer = optimizer
        self.trainer = trainer

        # Maybe put these in ordered dict
        self._skip_zero_grad: bool = False
        self._skip_backward: bool = False

    def run(self, epoch_idx: int) -> Tensor:
        closure = self._make_closure(epoch_idx)

        result = closure.closure()

        return result.loss

    def _make_closure(self, epoch_idx):
        step_fn = self._make_step_fn()
        backward_fn = self._make_backward_fn()
        zero_grad_fn = self._make_zero_grad_fn()

        return Closure(step_fn, backward_fn, zero_grad_fn)

    def _make_step_fn(self, epoch_idx) -> Callable[[], _ClosureResult]:
        with self.trainer.train_profiler.profile("Trainer forward step"):
            return self.trainer.training_step()

    def _make_backward_fn(self, loss: Tensor):
        if not self._skip_backward:
            with self.trainer.optimization_profiler.profile(
                "Optimizer backward step", self.optim_progress.step_idx
            ):
                return self.trainer.optimizer_step(loss)

        return None

    def _make_zero_grad_fn(self):
        if not self._skip_zero_grad:

            def zero_grad_fn(optimizer) -> None:
                with self.trainer.profiler.profile(
                    "Optimizer zero grad step.", self.optim_progress.step_idx
                ):
                    self.trainer.zero_grad_step(optimizer)

            return partial(zero_grad_fn, optimizer=self.optimizer)

        return None
