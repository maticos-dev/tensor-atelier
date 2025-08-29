from typing import Any, Dict, List, Optional, Union
import torch
from torch.optim import Optimizer


class AtelierOptimizer:
    """Wrapper for PyTorch optimizers with additional functionality."""

    def __init__(self, optimizer: Optimizer):
        self._optimizer = optimizer
        self._step_count = 0

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        self._step_count += 1
        return self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clear the gradients of all optimized tensors."""
        self._optimizer.zero_grad(set_to_none)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a param group to the optimizer's param_groups."""
        self._optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer state."""
        self._optimizer.load_state_dict(state_dict)

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state."""
        return self._optimizer.state_dict()

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return the optimizer's parameter groups."""
        return self._optimizer.param_groups

    @property
    def state(self) -> Dict[int, Dict[str, Any]]:
        """Return the optimizer's state."""
        return self._optimizer.state

    @property
    def step_count(self) -> int:
        """Return the number of optimization steps taken."""
        return self._step_count

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped optimizer."""
        return getattr(self._optimizer, name)
