import logging
from contextlib import contextmanager
from typing import override

from tensoratelier.profilers import BaseProfiler

log = logging.getLogger(__name__)


class _FittingProfiler(BaseProfiler):
    fitting_mode: str = "NO MODE SELECTED"
    fitting_epoch: int

    @override
    def start(self):
        log.debug(f"INITIATED [{self.fitting_mode.upper()}]; [{self.fitting_epoch}]")

    @override
    def stop(self):
        log.debug(f"COMPLETED [{self.fitting_mode.upper()}]; [{self.fitting_epoch}]")

    @override
    @contextmanager
    def profile(self, fitting_mode: str, fitting_epoch: int):
        self.fitting_mode = fitting_mode
        self.fitting_epoch = fitting_epoch

        try:
            self.start(fitting_mode)
            yield
        finally:
            self.stop(fitting_mode)
