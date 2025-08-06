import logging

from tensoratelier.core import AtelierTrainer
from tensoratelier.loops import _TrainingEpochLoop
from tensoratelier.loops.progress import _EpochProgress

log = logging.getLogger(__name__)


class _FitLoop:
    def __init__(self, trainer: AtelierTrainer, dataloader, max_epochs: int):
        if max_epochs <= 0:
            raise ValueError("Training requires at least one epoch")

        self.trainer = trainer

        self._data_source = dataloader
        self._epoch_progress = _EpochProgress()

        self.max_epochs = max_epochs
        self.epoch_loop = _TrainingEpochLoop()

    def setup_data(self):
        log.debug(f"{self.__class__.__name__} resetting dataloader")

        return self._data_source

    def on_advance_start(self):
        self._epoch_counter += 0.5

    def on_advance_end(self):
        self._epoch_counter += 0.5

        # # call train epoch end hooks
        # # we always call callback hooks first, but here we need to make an exception for the callbacks that
        # # monitor a metric, otherwise they wouldn't be able to monitor a key logged in
        # # `LightningModule.on_train_epoch_end`
        # call._call_callback_hooks(trainer, "on_train_epoch_end", monitoring_callbacks=False)
        # call._call_lightning_module_hook(trainer, "on_train_epoch_end")
        # call._call_callback_hooks(trainer, "on_train_epoch_end", monitoring_callbacks=True)

    def run(self):
        while not self.done:
            self.on_advance_start()
            self.advance()
            self.on_advance_end()

    def advance(self):
        log.debug("Initializing fit loop")

        source = self.setup_data()

        self.epoch_loop.run(self)

    @property
    def is_training_done(self) -> bool:
        assert self.epoch_counter <= self.max_epochs

        if self.epoch_counter == self.max_epochs:
            return True

        return False
