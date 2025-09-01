from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union


class BaseProfiler(ABC):
    """All user implementations of profilers.
    Must make user of this"""

    def __init__(self, **config: Any):
        """Initialize profiler with custom kwargs.

        Args:
            **kwargs: User-defined configuration options.
        """
        self.config = config
        self._active_profiles: Dict[str, Any] = {}

    @abstractmethod
    def start(self, desc: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def stop(self, desc: str, context: Any, **kwargs: Any) -> None:
        pass

    def profile(self, desc: str, **kwargs):
        # atelier doesnt worry about passing kwargs.
        # they are saved in baseprofiler at instantiation.
        return ProfilerContext(self, desc, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        self._active_profiles.clear()


class ProfilerContext:

    def __init__(self, profiler: BaseProfiler,
                 desc: str, **kwargs: Any):

        assert(isinstance(profiler, BaseProfiler))
        self.profiler = profiler 

        self.kwargs = kwargs
        self.desc = desc
        self.context = None

    def __enter__(self):
        self.context = self.profiler.start(self.desc, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop(self.desc, self.context, **self.kwargs)
