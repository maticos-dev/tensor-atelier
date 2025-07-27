class _StrategyRegistry:
    def __init__(self):
        self._registry = {}

    def _register(self, strategy_type: str):
        """
        this method will only be called
        at the definition of each class
        """
        pass


STRATEGY_REGISTRY = _StrategyRegistry()
