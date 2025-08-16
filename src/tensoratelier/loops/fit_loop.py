from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from tensoratelier.loops import _TrainingEpochLoop, _ValidationLoop
from tensoratelier.loops.progress import _EpochProgress

if TYPE_CHECKING:
    from tensoratelier.core import AtelierDataLoader, AtelierModule, AtelierTrainer

log = logging.getLogger(__name__)


class _FitLoop:
    def __init__(
        self,
        trainer: AtelierTrainer,
        model: AtelierModule,
        dataloader: AtelierDataLoader,
        max_epochs: int,
    ):
        if max_epochs <= 0:
            raise ValueError("Training requires at least one epoch")

        self.data_source = dataloader
        self.atelier_module = model

        self.max_epochs = max_epochs
        self.epoch_loop = _TrainingEpochLoop(trainer, 10)
        self.val_loop = _ValidationLoop()

        self._epoch_progress = _EpochProgress()

    def on_advance_start(self):
        self._epoch_progress.increment_started()

    def on_advance_end(self):
        self._epoch_progress.increment_completed()

    def run(self):
        while not self.done:
            self.on_advance_start()
            self.advance()
            self.on_advance_end()

    def advance(self):
        log.debug("Initializing fit loop")

        self.epoch_loop.run(self, self.atelier_module, self.data_source)

        # need mechanism to change the data source from the training to the validation segment.

        self.val_loop.run(self, self.atelier_module, self.data_source)

    @property
    def is_training_done(self) -> bool:
        assert self.epoch_counter <= self.max_epochs

        if self.epoch_counter == self.max_epochs:
            return True

        return False

    @property
    def done(self) -> bool:
        return self._epoch_progress.epoch_completed >= self.max_epochs
