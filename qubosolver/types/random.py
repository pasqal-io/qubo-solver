# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Random number generation utilities for QUBO solvers.

Provides a helper for creating `torch.Generator` instances that match the
device used for QUBO computations, so they can be passed directly to
tensor-sampling calls.
"""

from __future__ import annotations

from typing import Literal

import torch

from . import linalg


def torch_rng(seed: int | Literal["global"] | None = None) -> torch.Generator:
    """Creates a [`torch.Generator`][] compatible with [`qubosolver`][]'s torch typing.

    Args:
        seed: Optional seed for reproducibility. If ``None``, the generator
            is seeded from a non-deterministic source (a fresh
            `torch.Generator` otherwise always starts from the same fixed seed).
            If ``"global"``, the seed is drawn from the global PyTorch RNG,
            which advances it: after `torch.manual_seed(...)`, successive
            calls return different generators, and the sequence is reproducible.

    Returns:
        A `torch.Generator` instance, optionally seeded.
    """
    if seed == "global":
        seed = int(torch.randint(2**62, ()).item())
    generator = torch.Generator(linalg.device())
    if seed is None:
        generator.seed()
        return generator
    return generator.manual_seed(seed)
