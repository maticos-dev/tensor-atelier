from typing import Union

import torch
from tensorateier.strategies import Strategy

from tensoratelier.core import AtelierModule
from tensoratelier.core.connectors import AcceleratorHandler
from tensoratelier.utils import _wrap_args


class AtelierTrainer:
    @_wrap_args
    def __init__(
        self,
        *,
        max_epochs: int = 10,
        accelerator: Union[str, torch.device],
        strategy: Union[str, Strategy],
    ) -> None:
        self.max_epochs = max_epochs
        self._accelerator_handler = AcceleratorHandler(accelerator, strategy)

    def fit(self, model: AtelierModule, train_loader, validation_loader) -> None:
        # NB: some kind of type checking for train_loader, validation_loader, and model.
        # here error out the test script to see how these calls are made.

        self._accelerator_handler.move_model(model)
        self._accelerator_handler.move_dataloader(train_loader)
        self._acceleration_handler.move_dataloader(validation_loader)

        optimizer = model.configure_optimizers()

        for epoch in range(self.max_epochs):
            model.train()
            for batch in train_loader:
                batch = [x.to(self.device) for x in batch]
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                for batch in validation_loader:
                    batch = [x.to(self.device) for x in batch]
                    model.validation_step(batch)

    @property
    def strategy(self) -> Strategy:
        return self._accelerator_handler.strategy
