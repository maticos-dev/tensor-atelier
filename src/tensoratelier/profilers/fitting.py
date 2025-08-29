import logging
from contextlib import contextmanager
from typing import Any
from typing_extensions import override

from tensoratelier.profilers import BaseProfiler, ProfilerContext

log = logging.getLogger(__name__)


class _DefaultFittingProfiler(BaseProfiler):

    @override
    def start(self, desc: str, **kwargs: Any):
        log.debug(
            f"INITIATED [{desc.upper()}]; [{self.epoch_idx}]"
        )

    @override
    def stop(self, desc: str, **kwargs: Any):
        log.debug(
            f"COMPLETED [{desc.upper()}]; [{self.epoch_idx}]")

    @override
    def profile(self, desc: str, epoch_idx):
        self.epoch_idx = epoch_idx
        return ProfilerContext(self, desc, epoch_idx)
