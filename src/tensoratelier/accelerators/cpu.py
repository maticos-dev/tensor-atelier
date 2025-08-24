from typing import Union

import torch
from typing_extensions import override

from tensoratelier.accelerators import (
    ACCELERATOR_REGISTRY,
    BaseAccelerator,
)


@ACCELERATOR_REGISTRY._register("cpu")
class CPUAccelerator(BaseAccelerator):
    """Accelerator for CPU devices."""

    @override
    def setup(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup the model for CPU."""
        return model.to("cpu")

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "cpu":
            raise ValueError(f"Device should be CPU, got {device} instead.")

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str]) -> int:
        return _parse_cpu_cores(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, str]) -> list[torch.device]:
        devices = _parse_cpu_cores(devices)
        return [torch.device("cpu")] * devices

    @staticmethod
    @override
    def auto_device_count() -> int:
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        return True


def _parse_cpu_cores(cpu_cores: Union[int, str]) -> int:
    if isinstance(cpu_cores, str) and cpu_cores.strip().isdigit():
        cpu_cores = int(cpu_cores)

    if not isinstance(cpu_cores, int) or cpu_cores <= 0:
        raise TypeError(
            "'devices' selected with 'CPUAccelerator' should be an int > 0."
        )

    return cpu_cores
