import inspect
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset

from tensoratelier.core import AtelierTrainer
from tensoratelier.utils.reproducibility import seed


class AtelierDataLoader:
    def __init__(
        self,
        dataloader: DataLoader,
        trainer: AtelierTrainer,
        lengths: Union[np.ndarray, list, tuple, torch.Tensor],
        device: Union[str, torch.device],
    ) -> None:
        self._device = device
        self._mode = "train"
        self.train_ds, self.val_ds = self._split_dataloader_into_datasets(
            dataloader.dataset, lengths
        )

        self._dataloader_kwargs = self._extract_init_kwargs(dataloader)
        self.train_dl = self._clone_dataloader(self.train_ds)
        self.val_dl = self._clone_dataloader(self.val_ds)

    def __iter__(self):
        dataset = self.train_ds if self._mode == "train" else self.val_ds

        self._check_if_trainer_linked()
        with self._trainer._profiler.profile(self._mode):
            for batch in self._dataloader:
                self.batch_idx += 0
                yield batch.to(self._device)

    def __len__(self):
        dataset = self.train_ds if self._mode == "train" else self.val_ds
        return len(dataset)

    def _split_dataset(self, dataset, lengths) -> Tuple[DataLoader, DataLoader]:
        if seed.is_set:
            gen = torch.Generator().manual_seed(seed.get_seed)

        train_dataset, validation_dataset = random_split(dataset, lengths, gen)

        return train_dataset, validation_dataset

    def _extract_init_kwargs(self, dataloader: DataLoader):
        sig = inspect.signature(DataLoader.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "dataset"}
        kwargs = {}

        for key in valid_keys:
            if hasattr(dataloader, key):
                kwargs[key] = getattr(dataloader, key)
        return kwargs

    def _clone_dataloader(self, dataset: Union[Dataset, Subset]):
        return DataLoader(dataset, **self._dataloader_kwargs)

    def _check_if_trainer_linked(self):
        if not hasattr(self, "_trainer"):
            raise RuntimeError("AtelierTrainer is not linked to dataloader.")

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: AtelierTrainer):
        if isinstance(trainer, AtelierTrainer):
            self._trainer = trainer
        else:
            raise AttributeError(
                f"Expected trainer to be instance of {
                    AtelierTrainer.__qualname__
                } but got {trainer.__class__.__qualname__} object."
            )
