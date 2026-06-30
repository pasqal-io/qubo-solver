from __future__ import annotations

from typing import Any
import torch
from . import linalg


def dtype() -> torch.dtype:
    """Returns the globally configured float dtype."""
    return linalg.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(
    *args: Any,
    dtype: torch.dtype = dtype(),
    device: torch.device = device(),
    **kwargs: Any,
) -> linalg.Tensor:
    """Creates a zero-filled tensor with the given shape.

    Args:
        *args: Shape dimensions (e.g. ``zeros(2, 3)`` or ``zeros((2, 3))``).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to :func:`torch.zeros`.

    Returns:
        A tensor of zeros with the specified shape.
    """
    return torch.zeros(*args, dtype=dtype, device=device, **kwargs)


def tensor(
    data: Any,
    *,
    dtype: torch.dtype = dtype(),
    device: torch.device = device(),
    **kwargs: Any,
) -> linalg.Tensor:
    """Creates a tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to :func:`torch.tensor`.

    Returns:
        A tensor with the specified dtype and device.
    """
    return torch.tensor(data, dtype=dtype, device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> linalg.Tensor:
    """Converts an existing torch tensor to the global float dtype and device.

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to the global float dtype and device.
    """
    return tensor.to(dtype=dtype(), device=device())
