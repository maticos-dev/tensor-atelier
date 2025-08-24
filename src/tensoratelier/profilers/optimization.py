import logging
from contextlib import contextmanager
from typing import override

from tensoratelier.profilers import BaseProfiler

log = logging.getLogger(__name__)


class _OptimizationProfiler(BaseProfiler):
    optimization_policy: str = "dormant"
    step_idx: int = -1

    @override
    def start(self):
        log.debug(
            f"OPTIM INITIALIZING [{self.optimization_policy.upper()}]; STEP [{
                self.optimization_step
            }]"
        )

    @override
    def stop(self):
        log.debug(
            f"OPTIM COMPLETED [{self.optimization_policy.upper()}]; STEP [{
                self.optimization_step
            }]"
        )

    @override
    @contextmanager
    def profile(self, policy: str, idx: int):
        self.optimization_policy = policy
        self.optimization_step = idx

        try:
            self.start()
            yield
        finally:
            self.stop()
