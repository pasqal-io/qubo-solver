"""Square matrix utilities for QUBO solvers.

A [`Matrix`][qubosolver.Matrix] is a 2-D tensor of shape ``(n, n)`` using the globally configured
float dtype (float32 by default, float64 when double precision is enabled).
This module provides factory functions for creating and converting such matrices
on the globally configured torch device.

Typical usage:

    Q = matrix.zeros(4)                          # 4×4 zero matrix
    Q = matrix.tensor([[0, 1], [1, 0]])          # from nested list
    Q = matrix.as_tensor(some_tensor)            # cast existing tensor, no copy when possible
"""

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


def zeros(n: int, *, dtype: torch.dtype = dtype(), device: torch.device = device()) -> Matrix:
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


def as_tensor(data: Any) -> Matrix:
    """Convenience wrapper for `torch.as_tensor` that converts data to a matrix
    tensor, avoiding a copy when possible.

    If *data* is already a tensor with the right dtype and on the right device, it is
    returned as-is, sharing the same underlying memory. A numpy array is also shared
    rather than copied if it already has the global float dtype and the global device
    is ``cpu`` (numpy arrays only live on CPU, so any other dtype or device forces a
    copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, nested list, etc.).

    Returns:
        A 2-D tensor on the global dtype and device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())
