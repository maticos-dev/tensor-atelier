from typing import Type, Union

from tensoratelier.strategies import BaseStrategy


class _StrategyRegistry:
    def __init__(self):
        self._registry = {}

    def _register(self, strategy_type: str):
        """
        this method will only be called
        at the definition of each class
        """

        def wrapper(cls: Type[BaseStrategy]):
            if strategy_type.lower() in self._registry:
                raise ValueError("Strategy 'name' already registered")

                self._registry[strategy_type.lower()] = cls

        return wrapper

    def _get(self, strategy_type: Union[str, BaseStrategy]) -> BaseStrategy:
        key = strategy_type.lower()
        # NB: Need to develop the basestrategy class

        try:
            return self._registry[key]()

        except KeyError:
            raise ValueError(f"Unsupported strategy '{key}'.")


STRATEGY_REGISTRY = _StrategyRegistry()
