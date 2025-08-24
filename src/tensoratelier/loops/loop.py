from abc import ABC, abstractmethod
from typing import Any


class _Loop(ABC):
    """Base class for all training loops."""
    
    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._done = False
    
    @abstractmethod
    def run(self) -> None:
        """Run the loop."""
        pass
    
    @property
    def done(self) -> bool:
        """Check if the loop is done."""
        return self._done
    
    @done.setter
    def done(self, value: bool) -> None:
        self._done = value
