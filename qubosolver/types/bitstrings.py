"""Batch bitstring utilities for QUBO solvers.

A *bitstrings* collection is a 2-D ``torch.int8`` tensor of shape
``(count, n_bits)``, where each row is an individual bitstring.
This module provides factory functions and converters for creating and
manipulating batches of bitstrings on the globally configured torch device.

Typical usage:

    bs = bitstrings.from_strings(["1010", "0110", "1100"])
    ss = bitstrings.to_strings(bs)          # ["1010", "0110", "1100"]
    z  = bitstrings.zeros(4, 8)             # 4 zero bitstrings of length 8

See also qubosolver.types.bitstring for single-bitstring operations.
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from .linalg import Bitstrings
from . import bitstring


def dtype() -> torch.dtype:
    """Returns the dtype used for bitstrings (``torch.int8``)."""
    return bitstring.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return bitstring.device()


def zeros(count: int, n_bits: int, *, device: torch.device = device()) -> Bitstrings:
    """Creates a zero-filled 2-D bitstrings tensor.

    Args:
        count: Number of bitstrings (rows).
        n_bits: Length of each bitstring (columns).
        device: Torch device for the tensor.

    Returns:
        A 2-D ``int8`` tensor of shape ``(count, n_bits)``.
    """
    return torch.zeros((count, n_bits), dtype=dtype(), device=device)


def tensor(data: Any, *, device: torch.device = device(), **kwargs: Any) -> Bitstrings:
    """Creates a 2-D bitstrings tensor from the given data.

    Args:
        data: Input data (nested list or array-like of 0s and 1s).
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to `torch.tensor`.

    Returns:
        A 2-D ``int8`` tensor.
    """
    return torch.tensor(data, dtype=dtype(), device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> Bitstrings:
    """Converts an existing torch tensor to bitstrings (``int8``, on the global device).

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to ``int8`` on the global device.
    """
    return tensor.to(dtype=dtype(), device=device())


def from_strings(strings: Sequence[str], *, device: torch.device = device()) -> Bitstrings:
    """Creates a 2-D bitstrings tensor from a sequence of '0'/'1' strings.

    Args:
        strings: A sequence of strings, each consisting of '0' and '1' characters.
            All strings must have the same length.
        device: Torch device for the tensor.

    Returns:
        A 2-D ``int8`` tensor of shape ``(len(strings), len(strings[0]))``,
        or shape ``(0, 0)`` if *strings* is empty.

    Raises:
        ValueError: If the strings have differing lengths.
    """
    if len(strings) == 0:
        return zeros(0, 0, device=device)
    lengths = {len(s) for s in strings}
    if len(lengths) != 1:
        raise ValueError(
            f"All bitstrings must have the same length, got lengths: {sorted(lengths)}"
        )
    return torch.stack([bitstring.from_string(s, device=device) for s in strings])


def to_strings(bitstrings: Bitstrings) -> list[str]:
    """Converts a 2-D bitstrings tensor into a list of '0'/'1' strings.

    Args:
        bitstrings: A 2-D ``int8`` tensor of shape ``(n, m)`` containing 0s and 1s.

    Returns:
        A list of *n* strings, each of length *m*, representing each row of the tensor.
    """
    return [bitstring.to_string(b) for b in bitstrings]
