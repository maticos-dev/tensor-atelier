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
    @wraps(func)
    def wrapped(self, train_loader, validation_loader, *args, **kwargs):
        train_loader = _DeviceLoaderHandler(
            dataloader=train_loader, accelerator=self.accelerator
        )
        validation_loader = _DeviceLoaderHandler(
            dataloader=validation_loader, accelerator=self.accelerator
        )
        return func(train_loader, validation_loader, *args, **kwargs)

    return wrapped
