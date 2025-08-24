import torch
from typing import Any


class _ValidationLoop:
    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._done = False
    
    def run(self, model: torch.nn.Module, val_segment: Any) -> None:
        model.eval()
        with torch.no_grad():
            for batch in val_segment:
                model.validation_step(batch)
        self._done = True

    @property
    def done(self) -> bool:
        return self._done
