from typing import Any, Dict, List, Optional, Union
import torch


def get_batch_size(batch: Any) -> int:
    """Extract batch size from a batch of data."""
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    elif isinstance(batch, (list, tuple)):
        return get_batch_size(batch[0])
    elif isinstance(batch, dict):
        return get_batch_size(next(iter(batch.values())))
    else:
        raise ValueError(f"Cannot determine batch size for type {type(batch)}")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move a batch of data to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    elif isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    else:
        return batch


def is_tensor_or_tensor_list(obj: Any) -> bool:
    """Check if an object is a tensor or contains tensors."""
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, (list, tuple)):
        return any(is_tensor_or_tensor_list(item) for item in obj)
    elif isinstance(obj, dict):
        return any(is_tensor_or_tensor_list(value) for value in obj.values())
    else:
        return False
