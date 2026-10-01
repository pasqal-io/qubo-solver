"""Square matrix utilities for QUBO solvers.

A [`Matrix`][qubosolver.Matrix] is a 2-D tensor of shape ``(n, n)`` using the globally configured
float dtype (float32 by default, float64 when double precision is enabled).
This module provides factory functions for creating and converting such matrices
on the globally configured torch device.

Typical usage:

    Q = matrix.zeros(4)                          # 4x4 zero matrix
    Q = matrix.tensor([[0, 1], [1, 0]])          # from nested list
    Q = matrix.as_tensor(some_tensor)            # cast existing tensor, no copy when possible
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, overload

import torch

from qubosolver._checks import no_runtime_typecheck

from . import linalg
from .linalg import Matrix


def dtype() -> torch.dtype:
    """Returns the globally configured float dtype."""
    return linalg.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


_dtype = dtype  # alias so shadowed `dtype` params can still call the module function
_device = device  # alias so shadowed `device` params can still call the module function


@overload
def zeros(n: int, *, dtype: None = None, device: torch.device | None = None) -> Matrix: ...


@overload
def zeros(n: int, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor: ...


def zeros(
    n: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor:
    """Creates a zero-filled square matrix of shape ``(n, n)``.

    Args:
        n: Size of the matrix (number of rows and columns).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.

    Returns:
        A 2-D tensor of zeros with shape ``(n, n)``.
    """
    dtype = dtype or _dtype()
    device = device or _device()
    return torch.zeros((n, n), dtype=dtype, device=device)


@overload
def tensor(
    data: Any,  # noqa: ANN401 (array-like input forwarded to torch.tensor)
    *,
    dtype: None = None,
    device: torch.device | None = None,
    **kwargs: Any,  # noqa: ANN401 (forwarded to torch.tensor)
) -> Matrix: ...


@overload
def tensor(
    data: Any,  # noqa: ANN401 (array-like input forwarded to torch.tensor)
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    **kwargs: Any,  # noqa: ANN401 (forwarded to torch.tensor)
) -> torch.Tensor: ...


def tensor(
    data: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Creates a matrix tensor from the given data.

    Args:
        data: Input data (nested list or 2-D array-like).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 2-D tensor.
    """
    dtype = dtype or _dtype()
    device = device or _device()
    result = torch.tensor(data, dtype=dtype, device=device, **kwargs)
    # `torch.tensor([])` is 1-D: keep an empty matrix 2-D.
    return result.reshape(0, 0) if result.shape == (0,) else result


def as_tensor(data: Any) -> Matrix:  # noqa: ANN401 (array-like input forwarded to torch.as_tensor)
    """Convenience wrapper for `torch.as_tensor` that converts data to a matrix tensor.

    Avoids a copy when possible. If *data* is already a tensor with the right dtype and on
    the right device, it is returned as-is, sharing the same underlying memory. A numpy
    array is also shared rather than copied if it already has the global float dtype and
    the global device is ``cpu`` (numpy arrays only live on CPU, so any other dtype or
    device forces a copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, nested list, etc.).

    Returns:
        A 2-D tensor on the global dtype and device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())


# Returns a dataclass `Field`, typed as the tensor for static type checkers only.
@no_runtime_typecheck
def zeros_field(
    n: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Matrix:
    """Creates a dataclass field defaulting to a zero-filled square matrix.

    Args:
        n: Size of the matrix (number of rows and columns).
        dtype: Data type of the tensor.
        device: Torch device for the tensor.

    Returns:
        A dataclass field (typed as `Matrix` for the enclosing class) whose
        `default_factory` builds a fresh zero tensor per instance.
    """
    return field(default_factory=lambda: zeros(n, dtype=dtype, device=device))
