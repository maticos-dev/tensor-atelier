from abc import ABC, abstractmethod
from typing import Any

import torch

class Accelerator(ABC):
    '''
    Accelerator base class. Defines abstract functionality for dealing with one type of hardware. 
    '''
    @abstractmethod
    def setup_device(self, device: torch.device) -> None:
        '''Create and prep. device for current process'''

    @abstractmethod
    def teardown(self) -> None:
        '''Cleanup any state created by the accelerator'''

    @staticmethod
    @abstractmethod
    def parse_devices(devices: Any) -> Any:
        '''Accelerator device parsing logic'''

    @staticmethod
    @abstractmethod
    def get_parallel_devices(devices: Any) -> Any:
        '''Get parallel devices for the Accelerator'''

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        '''Get the device count when set to auto.'''

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        '''Detect if the hardware is available.'''