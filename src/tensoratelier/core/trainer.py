from typing import Union

import torch

from tensoratelier.core import AtelierModule
from tensoratelier.utils import _wrap_args


class AtelierTrainer:
    @_wrap_args
    def __init__(
        self, *, max_epochs: int = 10, device: Union[str, torch.device]
    ) -> None:
        self.max_epochs = max_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, model: AtelierModule, train_loader, validation_loader) -> None:
        model.to(self.device)
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
