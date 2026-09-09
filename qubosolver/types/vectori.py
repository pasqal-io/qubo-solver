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
    v = vectori.as_tensor(some_tensor)       # cast existing tensor to int64, no copy when possible

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


def as_tensor(data: Any) -> Vectori:
    """Convenience wrapper for `torch.as_tensor` that converts data to an integer
    vector tensor, avoiding a copy when possible.

    If *data* is already a tensor with the right dtype and on the right device, it is
    returned as-is, sharing the same underlying memory. A numpy array is also shared
    rather than copied if it already has ``int64`` dtype and the global device is
    ``cpu`` (numpy arrays only live on CPU, so any other dtype or device forces a
    copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, list, tuple, etc.).

    Returns:
        A 1-D ``int64`` tensor on the global device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())
