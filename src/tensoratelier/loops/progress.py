from dataclasses import dataclass


@dataclass
class _BatchProgress:
    step: int = 0


@dataclass
class _EpochProgress:
    epoch: int = 0
