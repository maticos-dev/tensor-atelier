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
        log.debug(
            f"INITIATED [{self.fitting_mode.upper()}]; [{self.fitting_epoch}]")

    @override
    def stop(self):
        log.debug(
            f"COMPLETED [{self.fitting_mode.upper()}]; [{self.fitting_epoch}]")

    @override
    @contextmanager
    def profile(self, policy: str, idx: int):
        self.fitting_mode = policy
        self.fitting_epoch = idx

        try:
            self.start()
            yield
        finally:
            self.stop()
