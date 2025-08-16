from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from tensoratelier.loops.optimization import _AutomaticOptimization, _StatefulBase
from tensoratelier.loops.progress import _BatchProgress, _OptimizationProgress

if TYPE_CHECKING:
    from tensoratelier.core import AtelierDataLoader, AtelierModule, AtelierTrainer
    from torch import Tensor


class _TrainingEpochLoop(_StatefulBase):
    def __init__(self, trainer: AtelierTrainer, max_steps: int):
        super().__init__(trainer)

        self.trainer = trainer
        self.max_steps = max_steps
        self.automatic_optimization = _AutomaticOptimization(
            trainer, self.trainer.atelier_optimizer
        )

        self._epoch_progress = self.trainer.epoch_progress
        self._optim_progress = _OptimizationProgress()
        self._batch_progress = _BatchProgress()

    def run(self, model: AtelierModule, data_loader: AtelierDataLoader):
        """
        Called once for each epoch. Only retrieve epoch once.
        Batches are not fixed, need to iteratively call
        batch progress.
        """

        epoch_idx = self._epoch_progress.epoch_idx

        self._update_current_state(epoch_idx, batch_idx=0)

        while not self.done:
            try:
                self.on_advance_start()
                self.advance(model, data_loader, epoch_idx)
                self.on_advance_end(epoch_idx)
            except StopIteration:
                break
            finally:
                self.on_iteration_end()

    def advance(
        self, model: AtelierModule, dataloader: AtelierDataLoader, epoch_idx: int
    ) -> Optional[Tensor]:
        self._update_current_state(epoch_idx, self._batch_progress.batch_idx)

        with self.trainer._profiler.profile("run_training_batch", epoch_idx):
            if self.trainer.atelier_module.automatic_optimization:
                loss = self.automatic_optimization.run(epoch_idx)
                return loss

    def on_advance_start(self):
        self._batch_progress.increment_started()

    def on_advance_end(self, epoch_idx: int):
        self._batch_progress.increment_completed()
        self._update_current_state(
            epoch_idx, batch_idx=self._batch_progress.batch_idx)

    def on_iteration_end(self):
        pass

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = self._is_max_steps_reached(
            self._optim_progress.step_idx, self.max_steps
        )

        return max_steps_reached

    # NB validation not yet implemented
    # @property
    # def _is_validation_done(self) -> bool:
    #     return self.val_loop.done

    def _is_max_steps_reached(self, current: int, maximum: int = 0) -> bool:
        return current >= maximum

    @property
    def _global_step(self) -> int:
        return self.automatic_optimization.optim_progress.step_idx

    @property
    def done(self) -> bool:
        # and self.is_validation_done (see ln. 72)
        return self._is_training_done
