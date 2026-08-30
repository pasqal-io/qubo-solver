"""Uniform random bitstring sampler for QUBO instances.

This module provides `random_sampling`, which samples uniformly random
binary vectors, evaluates their QUBO cost, and returns a deduplicated
`qubosolver.Solution` sorted by ascending cost.
"""

from __future__ import annotations

import torch

from qubosolver.types import Instance, Solution, bitstring, torch_rng
from qubosolver.utils import _costs


def solve(
    instance: Instance,
    *,
    max_bitstrings: int = 1,
    rng: torch.Generator = torch_rng(),
) -> Solution:
    """Sample uniformly random bitstring solutions for a QUBO instance.

    Draws `max_bitstrings` independent binary vectors uniformly at random,
    deduplicates them (identical samples are merged and their draw count is
    accumulated in `counts`), evaluates the QUBO cost of each unique
    bitstring.

    Note:
        Because of deduplication, the returned solution may contain fewer than
        `max_bitstrings` bitstrings when the same random vector is drawn more
        than once.

    Args:
        instance: The QUBO instance whose coefficient matrix is used to
            evaluate bitstring costs.
        max_bitstrings: Number of random bitstrings to draw before
            deduplication.  The returned solution may contain fewer unique
            bitstrings.
        rng: PyTorch random number generator controlling the sampling.

    Returns:
        A solution with unique bitstrings, their QUBO costs, draw counts, and probabilities.
    """
    bitstrings_ = bitstring.as_tensor(
        torch.randint(0, 2, size=(max_bitstrings, instance.size), generator=rng)
    )
    unique_bits, counts = torch.unique(bitstrings_, dim=0, return_counts=True)
    costs_ = _costs.batched_quadratic_cost(unique_bits.to(instance.matrix), instance.matrix)
    solution = (
        Solution(
            bitstrings=unique_bits,
            costs=costs_,
            counts=counts,
        )
        ._sort_by_cost()
        ._compute_probabilities()
    )

    return solution
