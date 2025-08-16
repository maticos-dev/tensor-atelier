import logging
from contextlib import contextmanager
from typing import override

from tensoratelier.profilers import BaseProfiler

log = logging.getLogger(__name__)


class _OptimizationProfiler(BaseProfiler):
    optimization_policy: str = None
    step_idx: str = None

    @override
    def start(self):
        log.debug(
            f"OPTIM INITIALIZING [{self.optimization_policy}]; STEP [{
                self.optimization_step
            }]"
        )

    @override
    def stop(self):
        log.debug(
            f"OPTIM COMPLETED [{self.optimization_policy}]; STEP [{
                self.optimization_step
            }]"
        )

    @override
    @contextmanager
    def profIle(self, optimization_policy: str, optimization_step: int):
        self.optimization_policy = optimization_policy
        self.optimization_step = optimization_step

        try:
            self.start()
            yield
        finally:
            self.stop()
