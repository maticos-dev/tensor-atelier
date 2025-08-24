from abc import ABC, abstractmethod
from typing import Any, Union

import torch


class BaseAccelerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup the model for this accelerator."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Teardown the accelerator."""
        pass


class Accelerator(BaseAccelerator):
    def __init__(self):
        super().__init__()
