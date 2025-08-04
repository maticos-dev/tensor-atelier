from typing import Union

import torch

from tensoratelier.core import AtelierModule
from tensoratelier.core.connectors import AcceleratorHandler
from tensoratelier.core.connectors.utils import auto_move_dataloader, auto_move_model
from tensoratelier.utils.parsing import _wrap_args


class AtelierTrainer:
    @_wrap_args
    def __init__(
        self,
        *,
        max_epochs: int = 10,
        accelerator: Union[str, torch.device],
    ) -> None:
        self.max_epochs = max_epochs
        self._accelerator_handler = AcceleratorHandler(accelerator)

    @auto_move_dataloader
    @auto_move_model
    def fit(self, model: AtelierModule, train_loader, validation_loader) -> None:
        # NB: some kind of type checking for train_loader, validation_loader, and model.
        # here error out the test script to see how these calls are made.

        optimizer = model.configure_optimizers()
        self.validate_optimizer(optimizer)

        for epoch in range(self.max_epochs):
            model.train()
            for batch in train_loader:
                loss = model.train_step(batch)

                self.backprop_and_reset_grads(loss, optimizer)

            model.eval()
            with torch.no_grad():
                for batch in validation_loader:
                    batch = [x.to(self.device) for x in batch]
                    model.validation_step(batch)

    def backprop_and_reset_grads(self, loss, optimizer):
        """
        Loss returned from the model.train_step is a number, not the loss
        class inself.
        """

        # if not isinstance(optimizer, torch.optim.Optimizer):
        #     raise TypeError(
        #         f"Expected type 'torch.optim.optimizer' but got {optimizer.__class__.__name__}"
        #     )

        if not isinstance(loss, torch.nn.Module):
            raise TypeError(
                f"Expected type 'torch.nn.Module' but got {loss.__class__.__name__}"
            )
            optimizer.step()
            optimizer.zero_grad()
            loss.backward()
