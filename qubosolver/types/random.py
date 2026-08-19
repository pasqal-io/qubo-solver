"""Random number generation utilities for QUBO solvers.

Provides a helper for creating `torch.Generator` instances on the globally
configured torch device.
"""

from __future__ import annotations

import torch

from . import linalg


def torch_rng(seed: int | None = None) -> torch.Generator:
    """Creates a `torch.Generator` on the global device.

    Args:
        seed: Optional seed for reproducibility. If ``None``, the generator
            is left with its default (non-deterministic) state.

    Returns:
        A `torch.Generator` instance, optionally seeded.
    """
    generator = torch.Generator(linalg.device())
    if seed is None:
        return generator
    return generator.manual_seed(seed)
