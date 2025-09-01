import logging
from contextlib import contextmanager
from typing import Any
from typing_extensions import override

from tensoratelier.profilers import BaseProfiler, ProfilerContext

log = logging.getLogger(__name__)


class _DefaultFittingProfiler(BaseProfiler):

    def __init__(self):
        super().__init__()

    @override
    def start(self, desc: str, **kwargs: Any):
        log.debug(
            f"INITIATED [{desc.upper()}]; [{kwargs['epoch_idx']}]"
        )

    @override
    def stop(self, desc: str, context: Any, **kwargs: Any):
        log.debug(
            f"COMPLETED [{desc.upper()}]; [{kwargs['epoch_idx']}]")

    @override
    def profile(self, desc: str, **kwargs):
        self.epoch_idx = kwargs.get("epoch_idx", None)
        return ProfilerContext(self, desc, epoch_idx = self.epoch_idx)
