from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module

class AttributeOverrideMixin:
    """Mixin to handle attribute name conflicts with parent classes."""
    
    # Define attribute mappings at class level
    _attribute_overrides: Dict[str, str] = {
        'optimizer': '_optimizer',
        # Add other conflicting attributes here
        # 'trainer': '_atelier_trainer',
        # 'device': '_atelier_device',
    }
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to handle conflicting attribute names."""
        # Access class-level attribute overrides
        overrides = getattr(self.__class__, '_attribute_overrides', {})
        
        if name in overrides:
            # Use the mapped private attribute name
            private_name = overrides[name]
            object.__setattr__(self, private_name, value)
        else:
            # Use normal setattr for everything else
            super().__setattr__(name, value)
    
    def __getattribute__(self, name: str) -> Any:
        """Override getattribute to handle mapped attributes."""
        try:
            # Get class-level overrides
            overrides = super(AttributeOverrideMixin, self).__getattribute__('_attribute_overrides')
            if overrides and name in overrides:
                private_name = overrides[name]
                return object.__getattribute__(self, private_name)
        except AttributeError:
            # _attribute_overrides doesn't exist or private_name doesn't exist
            pass
        
        # For everything else (including 'parameters'), use normal access
        return super().__getattribute__(name)
