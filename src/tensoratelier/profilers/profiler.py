from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager


class BaseProfiler(ABC):
    @abstractmethod
    def start(self, action_name: str = "") -> None:
        pass

    @abstractmethod
    def stop(self, action_name: str = "") -> None:
        pass

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        try:
            self.start(action_name)
            yield action_name

        finally:
            self.stop(action_name)
