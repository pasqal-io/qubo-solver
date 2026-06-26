from __future__ import annotations

from typing import Any
import torch
from . import linalg, vector


def dtype() -> torch.dtype:
    """Returns the dtype used for integer vectors (``torch.int64``)."""
    return torch.int64


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(n: int, *, device: torch.device = device()) -> linalg.Vectori:
    """Creates a zero-filled integer vector of length *n*.

    Args:
        n: Length of the vector.
        device: Torch device for the tensor.

    Returns:
        A 1-D ``int64`` tensor of zeros.
    """
    return vector.zeros(n, dtype=dtype(), device=device)


def tensor(data: Any, *, device: torch.device = device(), **kwargs: Any) -> linalg.Vectori:
    """Creates an integer vector tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like of integers).
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to :func:`torch.tensor`.

    Returns:
        A 1-D ``int64`` tensor.
    """
    return vector.tensor(data, dtype=dtype(), device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> linalg.Vectori:
    """Converts an existing torch tensor to an integer vector (``int64``, on the global device).

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to ``int64`` on the global device.
    """
    return tensor.to(dtype=dtype(), device=device())
