from tensoratelier.loops.progress import _BatchProgress
from tensoratelier.loops.utils import _is_max_steps_reached


class _TrainingEpochLoop:
    def __init__(self, max_steps: int):
        # self.validation_loop = _EvaluationLoop()
        self.batch_progress = _BatchProgress()
        self.max_steps = max_steps

    @property
    def total_batch_idx(self) -> int:
        return

    @property
    def _is_epoch_done(self) -> bool:
        max_steps_reached = _is_max_steps_reached(
            self.batch_progress.step, self.max_steps
        )

    @property
    def _global_step(self) -> int:
        return self.batch_progress.step
