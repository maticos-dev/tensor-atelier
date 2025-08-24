from __future__ import annotations
from abc import ABC
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tensoratelier.core import AtelierTrainer


class _StatefulBase(ABC):
    def __init__(self, trainer: AtelierTrainer) -> None:
        self.trainer = trainer
        self._restarting = False
        self._loaded_from_state_dict = False
        self._resuming_from_checkpoint = False

        # Current state tracking
        self._current_epoch_idx: Optional[int] = None
        self._current_batch_idx: Optional[int] = None

    @property
    def current_epoch_idx(self) -> Optional[int]:
        return self._current_epoch_idx

    @property
    def current_batch_idx(self) -> Optional[int]:
        return self._current_batch_idx

    @property
    def restarting(self) -> bool:
        return self._restarting

    @restarting.setter
    def restarting(self, restarting: bool) -> None:
        self._restarting = restarting
        for child in vars(self).values():
            if isinstance(child, _StatefulBase):
                child.restarting = restarting

    @property
    def is_resuming(self) -> bool:
        return self._resuming_from_checkpoint

    def _update_current_state(self, epoch_idx: int, batch_idx: int) -> None:
        self._current_batch_idx = batch_idx
        self._current_epoch_idx = epoch_idx

    def reset_restart_stage(self) -> None:
        pass

    def on_save_checkpoint(self) -> Dict[str, Any]:
        return {
            "current_epoch_idx": self._current_epoch_idx,
            "current_batch_idx": self._current_batch_idx,
        }

    def on_load_checkpoint(self, state_dict: Dict[str, Any]) -> None:
        self.current_batch_idx = state_dict.get("current_batch_idx")
        self.current_epoch_idx = state_dict.get("current_epoch_idx")

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = ""
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        for k, v in self.__dict__.items():
            key = prefix + k
            if isinstance(v, _StatefulBase):
                v.state_dict(destination, key + ".")

        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        self._load_from_state_dict(state_dict.copy(), prefix)

        for k, v in self.__dict__.items():
            if isinstance(v, _StatefulBase):
                v.load_state_dict(state_dict.copy(), prefix + k + ".")

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str) -> None:
        state_key = prefix + "state_dict"
        if state_key in state_dict:
            self.on_load_checkpoint(state_dict[state_key])

    def on_iteration_done(self) -> None:
        self._restarting = False
        self._loaded_from_state_dict = False
        self._resuming_from_checkpoint = False
        self.reset_restart_stage()
