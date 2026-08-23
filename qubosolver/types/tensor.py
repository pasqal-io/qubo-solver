"""Arbitrary-rank tensor utilities for QUBO solvers.

A [`Tensor`][qubosolver.Tensor] here is an arbitrary-rank float tensor using the globally configured
dtype (float32 by default, float64 when double precision is enabled).
This module provides factory functions for creating and converting such tensors
on the globally configured torch device.

Typical usage:

    t = tensor.zeros(2, 3)                   # 2×3 zero tensor
    t = tensor.tensor([[1.0, 0.0], [0.0, 1.0]])  # from nested list
    t = tensor.as_tensor(some_tensor)        # cast existing tensor, no copy when possible

For rank-specific aliases see [`qubosolver.vector`][qubosolver.vector] (1-D) and
[`qubosolver.matrix`][qubosolver.matrix] (2-D square).
"""

from __future__ import annotations

from typing import Any
import torch
from . import linalg
from .linalg import Tensor


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
) -> Tensor:
    """Creates a zero-filled tensor with the given shape.

    Args:
        *args: Shape dimensions (e.g. ``zeros(2, 3)`` or ``zeros((2, 3))``).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.zeros`.

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
) -> Tensor:
    """Creates a tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A tensor with the specified dtype and device.
    """
    return torch.tensor(data, dtype=dtype, device=device, **kwargs)


def as_tensor(data: Any) -> Tensor:
    """Convenience wrapper for `torch.as_tensor` that converts data to a tensor,
    avoiding a copy when possible.

    If *data* is already a tensor with the right dtype and on the right device, it is
    returned as-is, sharing the same underlying memory. A numpy array is also shared
    rather than copied if it already has the global float dtype and the global device
    is ``cpu`` (numpy arrays only live on CPU, so any other dtype or device forces a
    copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, list, tuple, etc.).

    Returns:
        A tensor on the global dtype and device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())
