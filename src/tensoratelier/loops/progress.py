from dataclasses import asdict, dataclass
from typing_extensions import Self


@dataclass
class _BaseProgress:
    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> Self:
        obj = cls()
        obj.load_state_dict(state_dict)
        return obj

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class _BatchProgress(_BaseProgress):
    batch_started: int = 0
    batch_completed: int = 0

    def increment_started(self):
        self.batch_started += 1

    def increment_completed(self):
        self.batch_completed += 1

    @property
    def batch_idx(self):
        return self.batch_completed - 1


@dataclass
class _EpochProgress(_BaseProgress):
    epoch_started: int = 0
    epoch_completed: int = 0

    def increment_started(self):
        self.epoch_started += 1

    def increment_completed(self):
        self.epoch_completed += 1

    @property
    def epoch_idx(self):
        return self.epoch_completed - 1


@dataclass
class _OptimizationProgress(_BaseProgress):
    step_started: int = 0
    step_completed: int = 0

    def increment_started(self):
        self.step_started += 1

    def increment_completed(self):
        self.step_completed += 1

    @property
    def step_idx(self):
        return self.step_completed - 1
