from contextlib import contextmanager
from typing import override

from tensoratelier.profilers import BaseProfiler


class _FittingProfiler(BaseProfiler):
    @override
    def start(self, fitting_mode: str):
        if fitting_mode.lower() in ("train", "validate"):
            return fitting_mode

        raise KeyError("Dataloader mode must be either 'train' or 'validate'")

    @override
    @contextmanager
    def profile(self, fitting_mode):
        try:
            self._mode = self.start(fitting_mode)
            yield
        except KeyError:
            pass
            # call debug log.
