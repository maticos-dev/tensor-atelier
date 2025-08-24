from torch import Tensor
from typing import Optional
from tensoratelier.loops.optimization import _AutomaticOptimization, _StatefulBase
from tensoratelier.loops.progress import _BatchProgress, _OptimizationProgress


class _TrainingEpochLoop(_StatefulBase):
    def __init__(self, trainer, max_steps: int):
        super().__init__(trainer)

        self.trainer = trainer
        self.max_steps = max_steps
        self.automatic_optimization = _AutomaticOptimization(
            trainer, self.trainer.atelier_optimizer
        )

        # Don't access epoch_progress here to avoid circular dependency
        self._epoch_progress = None
        self._optim_progress = _OptimizationProgress()
        self._batch_progress = _BatchProgress()

    def run(self, dataloader):
        """
        Called once for each epoch. Only retrieve epoch once.
        Batches are not fixed, need to iteratively call
        batch progress.
        """

        epoch_idx = self.epoch_progress.epoch_idx

        self._update_current_state(epoch_idx, batch_idx=0)

        while not self.done:
            try:
                self.on_advance_start()
                self.advance(dataloader, epoch_idx)
                self.on_advance_end()
            except StopIteration:
                break
            finally:
                self.on_iteration_end()

    def advance(self, dataloader, epoch_idx: int) -> Optional[Tensor]:
        self._update_current_state(epoch_idx, self._batch_progress.batch_idx)
        # need to get steps here.

        if self.trainer.atelier_module.automatic_optimization:
            loss = self.automatic_optimization.run()
            return loss

            # the loss can be returned as None from closureresult default value.

    def on_advance_start(self):
        self._batch_progress.increment_started()

    def on_advance_end(self):
        self._batch_progress.increment_completed()

    def on_iteration_end(self):
        pass

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = self._is_max_steps_reached(
            self._optim_progress.step_idx, self.max_steps
        )

        return max_steps_reached

    # @property
    # def _is_validation_done(self) -> bool:
    #     return self.val_loop.done

    def _is_max_steps_reached(self, current: int, maximum: int = 0) -> bool:
        return current >= maximum

    @property
    def _global_step(self) -> int:
        return self._optim_progress.step_idx

    @property
    def epoch_progress(self):
        if self._epoch_progress is None:
            self._epoch_progress = self.trainer.epoch_progress
        return self._epoch_progress

    @property
    def done(self) -> bool:
        return self._is_training_done  # and self.is_validation_done
