from __future__ import annotations
import inspect
from typing import Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset

from tensoratelier.utils.reproducibility import seed

if TYPE_CHECKING:
    from tensoratelier.core import AtelierTrainer


class AtelierDataLoader:
    def __init__(
        self,
        dataloader: DataLoader,
        trainer: AtelierTrainer,
        lengths: Union[np.ndarray, list, tuple, torch.Tensor],
        device: Union[str, torch.device],
    ) -> None:
        self.user_dataloader = dataloader
        self._device = device
        self._mode = "train"
        self.batch_idx = 0
        self.train_ds, self.val_ds = self._split_dataloader_into_subsets(
            self.user_dataloader.dataset, lengths
        )

        self._dataloader_kwargs = self._extract_init_kwargs(
            self.user_dataloader)
        self.train_dl = self._clone_dataloader(self.train_ds)
        self.val_dl = self._clone_dataloader(self.val_ds)

    def _split_dataloader_into_subsets(self,
                                       dataset,
                                       lengths) -> Tuple[Subset, Subset]:
        # Convert proportions to actual lengths if needed
        if isinstance(lengths, (list, tuple)) and all(isinstance(x, (int, float)) for x in lengths):
            if any(isinstance(x, float) for x in lengths):
                # Convert proportions to integers
                total_length = len(dataset)
                train_length = int(lengths[0] * total_length)
                val_length = total_length - train_length
                lengths = [train_length, val_length]
                # print(f"Dataset split: total={total_length}, train={train_length}, val={val_length}")

        # print(f"Using lengths: {lengths}")
        # For now, use random_split without generator to avoid potential issues
        train_dataset, validation_dataset = random_split(dataset, lengths)

        return train_dataset, validation_dataset

    def _extract_init_kwargs(self, dataloader: DataLoader):
        '''
            get initialization arguments for original
            dataloader object passed by user.

            recycle them in initialization of dataloader
            object here.
        '''
        sig = inspect.signature(DataLoader.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "dataset"}
        kwargs = {}

        for key in valid_keys:
            if hasattr(dataloader, key):
                kwargs[key] = getattr(dataloader, key)
        
        # Remove batch_sampler if present to avoid conflicts with batch_size, shuffle, etc.
        if 'batch_sampler' in kwargs:
            del kwargs['batch_sampler']
        
        # Remove sampler if present to avoid conflicts with dataset size
        if 'sampler' in kwargs:
            del kwargs['sampler']
        
        # Debug: print the extracted kwargs
        # print(f"Extracted DataLoader kwargs: {kwargs}")
            
        return kwargs

    def _clone_dataloader(self, dataset: Union[Dataset, Subset]):
        return DataLoader(dataset, **self._dataloader_kwargs)

    def _check_if_trainer_linked(self):
        if not hasattr(self, "_trainer"):
            raise RuntimeError("AtelierTrainer is not linked to dataloader.")

    def _is_generator_configurable(self):
        return seed.is_set

    def _configure_generator(self):
        gen = {'generator': torch.Generator().manual_seed(
            seed.get_seed)} if self._is_generator_configurable() else {}
        return gen

    def __iter__(self):
        dataset = self.train_dl if self._mode == "train" else self.val_dl

        self._check_if_trainer_linked()
        with self._trainer.train_profiler.profile(self._mode, self.batch_idx):
            for batch in dataset:
                self.batch_idx += 1
                # Move each tensor in the batch to the device
                if isinstance(batch, (list, tuple)):
                    yield [item.to(self._device) if hasattr(item, 'to') else item for item in batch]
                else:
                    yield batch.to(self._device) if hasattr(batch, 'to') else batch

    def __next__(self):
        if not hasattr(self, '_iterator'):
            self._iterator = iter(self)
            # returns dataloader generator obj.
            # iter called only once.
            # calls self.__iter__

        try:
            return next(self._iterator)
            # calls next batch in dataloader gen.
        except StopIteration:
            delattr(self, '_iterator')
            raise

    def forward(self):
        return next(self)  # calls self.__next__

    def __len__(self):
        return len(self.user_dataloader)

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        # Import here to avoid circular imports
        from tensoratelier.core import AtelierTrainer
        
        if isinstance(trainer, AtelierTrainer):
            self._trainer = trainer
        else:
            raise AttributeError(
                f"Expected trainer to be instance of {AtelierTrainer.__qualname__} "
                f"but got {trainer.__class__.__qualname__} object."
            )
