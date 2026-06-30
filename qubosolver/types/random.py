from __future__ import annotations

import torch

from . import linalg


def torch_rng(seed: int | None = None) -> torch.Generator:
    """Creates a :class:`torch.Generator` on the global device.

    Args:
        seed: Optional seed for reproducibility. If ``None``, the generator
            is left with its default (non-deterministic) state.

    Returns:
        A :class:`torch.Generator` instance, optionally seeded.
    """
    generator = torch.Generator(linalg.device())
    if seed is None:
        return generator
    return generator.manual_seed(seed)
