from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager


class BaseProfiler(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @contextmanager
    def profile(self, policy: str, idx: int) -> Generator:
        try:
            self.start()
            yield

        finally:
            self.stop()
