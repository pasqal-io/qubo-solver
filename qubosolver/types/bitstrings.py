"""Batch bitstring utilities for QUBO solvers.

A [`Bitstrings`][qubosolver.Bitstrings] collection is a 2-D ``torch.int8`` tensor of shape
``(count, n_bits)``, where each row is an individual bitstring.
This module provides factory functions and converters for creating and
manipulating batches of bitstrings on the globally configured torch device.

Typical usage:

    bs = bitstrings.from_strings(["1010", "0110", "1100"])
    ss = bitstrings.to_strings(bs)          # ["1010", "0110", "1100"]
    z  = bitstrings.zeros(4, 8)             # 4 zero bitstrings of length 8

See also [`qubosolver.bitstring`][qubosolver.bitstring] for single-bitstring operations.
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from .linalg import Bitstrings
from . import bitstring
from .random import torch_rng


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


def from_strings(strings: Sequence[str], *, device: torch.device = device()) -> Bitstrings:
    """Creates a 2-D bitstrings tensor from a sequence of '0'/'1' strings.

    Args:
        strings: A sequence of strings, each consisting of '0' and '1' characters.
            All strings must have the same length.
        device: Torch device for the tensor.

    Returns:
        A 2-D ``int8`` tensor of shape ``(len(strings), len(strings[0]))``, possibly empty.

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


def rand(
    count: int, n_bits: int, *, device: torch.device = device(), rng: torch.Generator = torch_rng()
) -> Bitstrings:
    """Creates a 2-D bitstrings tensor with independent uniformly random bits.

    Args:
        count: Number of bitstrings (rows).
        n_bits: Length of each bitstring (columns).
        device: Torch device for the tensor.
        rng: PyTorch random number generator controlling the sampling.

    Returns:
        A 2-D ``int8`` tensor of shape ``(count, n_bits)`` containing 0s and 1s.
    """
    return torch.randint(0, 2, (count, n_bits), generator=rng, device=device, dtype=dtype())


def as_tensor(data: Any) -> Bitstrings:
    """Convenience wrapper for `torch.as_tensor` that converts data to a bitstrings
    tensor, avoiding a copy when possible.

    If *data* is already a tensor with the right dtype and on the right device, it is
    returned as-is, sharing the same underlying memory. A numpy array is also shared
    rather than copied if it already has ``int8`` dtype and the global device is
    ``cpu`` (numpy arrays only live on CPU, so any other dtype or device forces a
    copy). Lists, tuples, and other array-like inputs are always copied.

    Args:
        data: Input data (tensor, numpy array, nested list, etc.).

    Returns:
        A 2-D ``int8`` tensor on the global device.
    """
    return torch.as_tensor(data, dtype=dtype(), device=device())
