# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Batch bitstring utilities for QUBO solvers.

A batch of bitstrings is a 2-D ``int8`` array of shape
``(count, n_bits)``, where each row is an individual bitstring.
This module provides converters for creating and manipulating batches of bitstrings.

Typical usage:

    bs = bitstrings.from_strings(["1010", "0110", "1100"])
    ss = bitstrings.to_strings(bs)          # ["1010", "0110", "1100"]
    f  = bitstrings.round([[1.0, 0.0, 0.9999999]])  # from a MIP solver's output

See also [`qubosolver.bitstring`][qubosolver.bitstring] for single-bitstring operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .random import numpy_rng

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def from_strings(strings: Sequence[str]) -> NDArray[np.int8]:
    """Creates a 2-D bitstrings array from a sequence of '0'/'1' strings.

    Args:
        strings: A sequence of strings, each consisting of '0' and '1' characters.
            All strings must have the same length.

    Returns:
        A 2-D ``int8`` array of shape ``(len(strings), len(strings[0]))``, possibly empty.

    Raises:
        ValueError: If the strings have differing lengths.
    """
    if len(strings) == 0:
        return np.zeros((0, 0), dtype=np.int8)
    lengths = {len(s) for s in strings}
    if len(lengths) != 1:
        msg = f"All bitstrings must have the same length, got lengths: {sorted(lengths)}"
        raise ValueError(msg)
    return np.asarray([[int(c) for c in s] for s in strings], dtype=np.int8)


def round(  # noqa: A001 (public API, mirrors builtins.round)
    data: Any,  # noqa: ANN401 (array-like input forwarded to np.asarray)
    *,
    atol: float = 1e-6,
) -> NDArray[np.int8]:
    """Rounds near-integral float values to a 2-D bitstrings array.

    Args:
        data: Input data (numpy array, nested list, etc.) of floats, each
            within *atol* of 0 or 1. Nested sequences must not be ragged.
        atol: Maximum absolute distance from 0 or 1 tolerated before raising.

    Returns:
        A 2-D ``int8`` array of shape ``(count, n_bits)``.

    Raises:
        ValueError: If any value is further than *atol* from both 0 and 1, or if
            the input is a ragged nested sequence.
    """
    values = np.asarray(data, dtype=np.float64)
    bits = np.round(values)
    invalid = ((bits != 0) & (bits != 1)) | (np.abs(values - bits) > atol)
    if bool(invalid.any()):
        msg = (
            f"Expected values within {atol} of 0 or 1, got "
            f"{values[invalid].tolist()} in {values.tolist()}"
        )
        raise ValueError(msg)
    if bits.shape == (0,):
        # `np.asarray([])` is 1-D: keep empty bitstrings 2-D.
        bits = bits.reshape(0, 0)
    return bits.astype(np.int8)


def to_strings(bitstrings: NDArray[np.int8]) -> list[str]:
    """Converts a 2-D bitstrings array into a list of '0'/'1' strings.

    Args:
        bitstrings: A 2-D ``int8`` array of shape ``(n, m)`` containing 0s and 1s.

    Returns:
        A list of *n* strings, each of length *m*, representing each row of the array.
    """
    return ["".join(str(b) for b in row.tolist()) for row in bitstrings]


def rand(count: int, n_bits: int, *, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
    """Creates a 2-D bitstrings array with independent uniformly random bits.

    Args:
        count: Number of bitstrings (rows).
        n_bits: Length of each bitstring (columns).
        rng: NumPy random number generator controlling the sampling.

    Returns:
        A 2-D ``int8`` array of shape ``(count, n_bits)`` containing 0s and 1s.
    """
    rng = rng or numpy_rng()
    return rng.integers(0, 2, (count, n_bits), dtype=np.int8)
