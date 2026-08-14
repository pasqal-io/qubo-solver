"""Bitstring utilities for QUBO solvers.

A [`Bitstring`][qubosolver.Bitstring] is a 1-D ``torch.int8`` tensor whose elements are 0 or 1.
This module provides factory functions and converters for creating and
manipulating bitstrings on the globally configured torch device.

Typical usage:

    bs = bitstring.from_string("1010")
    s  = bitstring.to_string(bs)        # "1010"
    z  = bitstring.zeros(4)             # tensor([0, 0, 0, 0], dtype=torch.int8)
"""

from __future__ import annotations

from typing import Any
import torch
from . import linalg, vector
from .linalg import Bitstring
from .random import torch_rng


def dtype() -> torch.dtype:
    """Returns the dtype used for bitstrings (``torch.int8``)."""
    return torch.int8


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return linalg.device()


def zeros(n: int, *, device: torch.device = device()) -> Bitstring:
    """Creates a zero-filled bitstring of length *n*.

    Args:
        n: Length of the bitstring.
        device: Torch device for the tensor.

    Returns:
        A 1-D ``int8`` tensor of zeros.
    """
    return vector.zeros(n, dtype=dtype(), device=device)


def tensor(data: Any, *, device: torch.device = device(), **kwargs: Any) -> Bitstring:
    """Creates a bitstring tensor from the given data.

    Args:
        data: Input data (list, tuple, or array-like of 0s and 1s).
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 1-D ``int8`` tensor.
    """
    return vector.tensor(data, dtype=dtype(), device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> Bitstring:
    """Converts an existing torch tensor to a bitstring (``int8``, on the global device).

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to ``int8`` on the global device.
    """
    return tensor.to(dtype=dtype(), device=device())


def from_string(s: str, *, device: torch.device = device()) -> Bitstring:
    """Creates a bitstring tensor from a string of '0' and '1' characters.

    Args:
        s: A string consisting of '0' and '1' characters.
        device: Torch device for the tensor.

    Returns:
        A 1-D ``int8`` tensor.
    """
    return tensor([int(c) for c in s], device=device)


def to_string(bitstring: Bitstring) -> str:
    """Converts a bitstring tensor to its string representation.

    Args:
        bitstring: A 1-D ``int8`` tensor of 0s and 1s.

    Returns:
        A string of '0' and '1' characters.
    """
    return "".join(str(b.item()) for b in bitstring.flatten())


def rand(
    n: int, *, device: torch.device = device(), rng: torch.Generator = torch_rng()
) -> Bitstring:
    """Creates a bitstring of length *n* with independent uniformly random bits.

    Args:
        n: Length of the bitstring.
        device: Torch device for the tensor.
        rng: PyTorch random number generator controlling the sampling.

    Returns:
        A 1-D ``int8`` tensor of 0s and 1s.
    """
    return torch.randint(0, 2, (n,), generator=rng, device=device, dtype=dtype())
