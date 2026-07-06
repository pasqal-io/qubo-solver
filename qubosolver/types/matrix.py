from __future__ import annotations

from typing import Any
import torch
from . import linalg
from .linalg import Matrix


def dtype() -> torch.dtype:
    """Returns the globally configured float dtype."""
    return linalg.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(
    n: int, *, dtype: torch.dtype = dtype(), device: torch.device = device()
) -> Matrix:
    """Creates a zero-filled square matrix of shape ``(n, n)``.

    Args:
        n: Size of the matrix (number of rows and columns).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.

    Returns:
        A 2-D tensor of zeros with shape ``(n, n)``.
    """
    return torch.zeros((n, n), dtype=dtype, device=device)


def tensor(
    data: Any,
    *,
    dtype: torch.dtype = dtype(),
    device: torch.device = device(),
    **kwargs: Any,
) -> Matrix:
    """Creates a matrix tensor from the given data.

    Args:
        data: Input data (nested list or 2-D array-like).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 2-D tensor.
    """
    return torch.tensor(data, dtype=dtype, device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> Matrix:
    """Converts an existing torch tensor to a matrix with the global dtype and device.

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to the global float dtype and device.
    """
    return tensor.to(dtype=dtype(), device=device())
