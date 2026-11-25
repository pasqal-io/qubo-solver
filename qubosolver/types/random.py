# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Random number generation utilities for QUBO solvers.

Provides a helper for creating `numpy.random.Generator` instances that can be passed
directly to the solvers' ``rng`` parameters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def numpy_rng(seed: int | Literal["global"] | None = None) -> np.random.Generator:
    """Creates a [`numpy.random.Generator`][] for the solvers' ``rng`` parameters.

    Args:
        seed: Optional seed for reproducibility. If ``None``, the generator
            is seeded from a non-deterministic source. If ``"global"``, the seed
            is drawn from NumPy's legacy global RNG, which advances it: after
            `numpy.random.seed(...)`, successive calls return different
            generators, and the sequence is reproducible.

    Returns:
        A `numpy.random.Generator` instance, optionally seeded.
    """
    if seed == "global":
        seed = int(np.random.randint(2**62, dtype=np.int64))
    return np.random.default_rng(seed)
