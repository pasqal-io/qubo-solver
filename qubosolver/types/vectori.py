"""1-D integer vector utilities for QUBO solvers.

A [`Vectori`][qubosolver.Vectori] is a 1-D ``torch.int64`` tensor of shape ``(n,)`` used to
represent integer-valued quantities such as indices or counts.
Unlike [`qubosolver.vector`][qubosolver.vector], the dtype is fixed to ``int64``
and is not affected by the global float precision setting.

This module provides factory functions for creating and converting integer
vectors on the globally configured torch device.

Typical usage:

    v = vectori.zeros(4)                     # 1-D zero int64 vector of length 4
    v = vectori.tensor([0, 1, 2, 3])         # from a list of integers
    v = vectori.from_torch(some_tensor)      # cast existing tensor to int64

See also [`qubosolver.vector`][qubosolver.vector] for float vectors.
"""

from __future__ import annotations

from typing import Any
import torch
from . import linalg, vector
from .linalg import Vectori


def dtype() -> torch.dtype:
    """Returns the dtype used for integer vectors (``torch.int64``)."""
    return torch.int64


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(n: int, *, device: torch.device = device()) -> Vectori:
    """Creates a zero-filled integer vector of length *n*.

    Args:
        n: Length of the vector.
        device: Torch device for the tensor.

    Returns:
        A 1-D ``int64`` tensor of zeros.
    """
    return vector.zeros(n, dtype=dtype(), device=device)


def tensor(data: Any, *, device: torch.device = device(), **kwargs: Any) -> Vectori:
    """Creates an integer vector tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like of integers).
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 1-D ``int64`` tensor.
    """
    return vector.tensor(data, dtype=dtype(), device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> Vectori:
    """Converts an existing torch tensor to an integer vector (``int64``, on the global device).

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to ``int64`` on the global device.
    """
    return tensor.to(dtype=dtype(), device=device())
