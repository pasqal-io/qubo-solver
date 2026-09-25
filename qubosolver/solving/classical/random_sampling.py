"""Uniform random bitstring sampler for QUBO instances."""

from __future__ import annotations

import torch

from qubosolver.types import Instance, Solution, bitstrings, torch_rng, vector, vectori


def solve(
    instance: Instance,
    *,
    max_bitstrings: int = 1,
    rng: torch.Generator | None = None,
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
    rng = rng or torch_rng()
    solution = Solution(
        bitstrings=bitstrings.rand(max_bitstrings, instance.size, rng=rng),
        costs=vector.zeros(max_bitstrings),
        counts=vectori.zeros(max_bitstrings).fill_(1),
        probabilities=vector.zeros(max_bitstrings),
    )
    solution = solution.deduplicate(update=False)._update(instance)

    return solution
