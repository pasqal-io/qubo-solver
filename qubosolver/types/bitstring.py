# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Bitstring utilities for QUBO solvers.

A bitstring is a 1-D ``int8`` array whose elements are 0 or 1.
This module provides converters for creating and manipulating bitstrings.

Typical usage:

    bs = bitstring.from_string("1010")
    s  = bitstring.to_string(bs)        # "1010"
    f  = bitstring.round([1.0, 0.0, 0.9999999])         # from a MIP solver's output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import bitstrings
from .random import numpy_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray


def from_string(s: str) -> NDArray[np.int8]:
    """Creates a bitstring array from a string of '0' and '1' characters.

    Args:
        s: A string consisting of '0' and '1' characters.

    Returns:
        A 1-D ``int8`` array.
    """
    result: NDArray[np.int8] = bitstrings.from_strings([s])[0]
    return result


def round(  # noqa: A001 (public API, mirrors builtins.round)
    data: Any,  # noqa: ANN401 (array-like input forwarded to np.asarray)
    *,
    atol: float = 1e-6,
) -> NDArray[np.int8]:
    """Rounds near-integral float values to a bitstring array.

    Args:
        data: Input data (numpy array, list, etc.) of floats, each
            within *atol* of 0 or 1.
        atol: Maximum absolute distance from 0 or 1 tolerated before raising.

    Returns:
        A 1-D ``int8`` array of 0s and 1s.

    Raises:
        ValueError: If any value is further than *atol* from both 0 and 1.
    """
    values = np.asarray(data, dtype=np.float64)
    result: NDArray[np.int8] = bitstrings.round(values[np.newaxis], atol=atol)[0]
    return result


def to_string(bitstring: NDArray[np.int8]) -> str:
    """Converts a bitstring array to its string representation.

    Args:
        bitstring: A 1-D ``int8`` array of 0s and 1s.

    Returns:
        A string of '0' and '1' characters.
    """
    return bitstrings.to_strings(bitstring.reshape(1, -1))[0]


def rand(n: int, *, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
    """Creates a bitstring of length *n* with independent uniformly random bits.

    Args:
        n: Length of the bitstring.
        rng: NumPy random number generator controlling the sampling.

    Returns:
        A 1-D ``int8`` array of 0s and 1s.
    """
    rng = rng or numpy_rng()
    return rng.integers(0, 2, (n,), dtype=np.int8)
