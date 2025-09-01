import logging
from contextlib import contextmanager
from typing_extensions import override

from tensoratelier.profilers import AtelierBaseProfiler

log = logging.getLogger(__name__)


class _OptimizationProfiler(AtelierBaseProfiler):
    optimization_policy: str = "dormant"
    step_idx: int = -1

    @override
    def start(self):
        log.debug(
            f"OPTIM INITIALIZING [{self.optimization_policy.upper()}]; "
     	    f"STEP [{self.optimization_step}]"
        )

    @override
    def stop(self):
        log.debug(
            f"OPTIM COMPLETED [{self.optimization_policy.upper()}];"
	    f"STEP [{self.optimization_step}]"
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
