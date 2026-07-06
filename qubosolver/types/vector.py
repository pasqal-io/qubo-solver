"""1-D vector utilities for QUBO solvers.

A [`Vector`][qubosolver.Vector] is a 1-D float tensor of shape ``(n,)`` using the globally configured
dtype (float32 by default, float64 when double precision is enabled).
This module provides factory functions for creating and converting such vectors
on the globally configured torch device.

Typical usage:

    v = vector.zeros(4)                      # 1-D zero vector of length 4
    v = vector.tensor([1.0, 0.5, -1.0])     # from a list
    v = vector.from_torch(some_tensor)       # cast existing tensor

For higher-rank variants see [`qubosolver.matrix`][qubosolver.matrix] (2-D square) and
[`qubosolver.tensor`][qubosolver.tensor] (arbitrary rank).
"""

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
