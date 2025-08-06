import inspect
from functools import wraps
from typing import Union

import torch


class _DeviceLoaderHandler:
    def __init__(
        self, dataloader: torch.data.utils.DataLoader, device: Union[str, torch.device]
    ) -> None:
        self._device = device
        self._dataloader = dataloader

    def __iter__(self):
        for batch in self._dataloader:
            return batch.to(self._device)

    def __len__(self):
        return len(self.dataloader)


def auto_move_dataloader(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        train_loader = list(bound.arguments.keys())["train_loader"]
        validation_loader = list(bound.arguments.keys())["validation_loader"]

        train_loader = _DeviceLoaderHandler(
            dataloader=train_loader, accelerator=self.accelerator
        )
        validation_loader = _DeviceLoaderHandler(
            dataloader=validation_loader, accelerator=self.accelerator
        )
        return func(train_loader, validation_loader, *args, **kwargs)

    return wrapped
