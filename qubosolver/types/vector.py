from __future__ import annotations

from typing import Any
import torch
from . import linalg
from .linalg import Vector


def dtype() -> torch.dtype:
    """Returns the globally configured float dtype."""
    return linalg.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(
    n: int, *, dtype: torch.dtype = dtype(), device: torch.device = device()
) -> Vector:
    """Creates a zero-filled 1-D vector of length *n*.

    Args:
        n: Length of the vector.
        dtype: Data type of the tensor.
        device: Torch device for the tensor.

    Returns:
        A 1-D tensor of zeros.
    """
    return torch.zeros(n, dtype=dtype, device=device)


def tensor(
    data: Any,
    *,
    dtype: torch.dtype = dtype(),
    device: torch.device = device(),
    **kwargs: Any,
) -> Vector:
    """Creates a 1-D vector tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 1-D tensor.
    """
    return torch.tensor(data, dtype=dtype, device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> Vector:
    """Converts an existing torch tensor to a vector with the global dtype and device.

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to the global float dtype and device.
    """
    return tensor.to(dtype=dtype(), device=device())
