from typing import Union

import torch

from tensoratelier.accelerators import BaseAccelerator, registry
from tensoratelier.core import AtelierModule
from tensoratelier.strategies import Strategy


class AcceleratorHandler:
    def __init__(
        self, accelerator: Union[str, torch.device], strategy: Union[str, Strategy]
    ):
        self._accelerator_flag: Union[BaseAccelerator, str] = "auto"
        self._strategy_flag: Union[Strategy, str] = "auto"

        self._check_strategy_and_accelerator_flags(
            accelerator=accelerator, strategy=strategy
        )

        # execution should not continue past here if any checks fail.
        if self._accelerator_flag == "auto":
            self._accelerator_flag = self._auto_select_accelerator()
        else:
            self._accelerator_flag = accelerator

    def move_dataloader(dataloader):
        if dataloader.collate_fn is None:
            # wrap somehow. dont interfere with user.
            pass

    def move_model(self, model: AtelierModule):
        model.to(self._accelerator_flag)

    def _check_strategy_and_accelerator_flag(
        accelerator: Union[str, torch.device], strategy: Union[Strategy, str]
    ):
        accelerator_cls = registry._get_accelerator(accelerator)

        if not accelerator_cls:
            raise ValueError(
                f"You selected an invalid accelerator name: `accelerator={
                    accelerator!r
                }`."
                f" Available names are: auto, {
                    ', '.join(registry._ACCELERATOR_REGISTRY)
                }."
            )

        # perform checking for strategy

    def _auto_select_accelerator():
        # as more accelerators are supported,
        # will need to check if any are available.
        # no need to check if cpu is available
        # or program would not be even running.
        return "cpu"

    def _init_strategy(self) -> None:
        """
        Instantiate the Strategy depending on setting of
        ''_strategy_flag''
        """

        assert isinstance(self._startegy_flag, (str, Strategy))
        if isinstance(self._strategy_flag, str):
            # self.strategy = StrategyRegistry.get(self._strategy_flag)
            pass
        else:
            self.strategy = self._strategy_flag
