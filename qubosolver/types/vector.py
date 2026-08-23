"""1-D vector utilities for QUBO solvers.

A [`Vector`][qubosolver.Vector] is a 1-D float tensor of shape ``(n,)`` using the globally configured
dtype (float32 by default, float64 when double precision is enabled).
This module provides factory functions for creating and converting such vectors
on the globally configured torch device.

Typical usage:

    v = vector.zeros(4)                      # 1-D zero vector of length 4
    v = vector.tensor([1.0, 0.5, -1.0])     # from a list
    v = vector.as_tensor(some_tensor)        # cast existing tensor, no copy when possible

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


def zeros(n: int, *, dtype: torch.dtype = dtype(), device: torch.device = device()) -> Vector:
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


def as_tensor(data: Any) -> Vector:
    """Convenience wrapper for `torch.as_tensor` that converts data to a vector
    tensor, avoiding a copy when possible.

    If *data* is already a tensor with the right dtype and on the right device, it is
    returned as-is, sharing the same underlying memory. A numpy array is also shared
    rather than copied if it already has the global float dtype and the global device
    is ``cpu`` (numpy arrays only live on CPU, so any other dtype or device forces a
    copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, list, tuple, etc.).

    Returns:
        A 1-D tensor on the global dtype and device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())
