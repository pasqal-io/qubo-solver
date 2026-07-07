"""Uniform random bitstring sampler for QUBO instances.

This module provides `random_solutions`, which samples uniformly random
binary vectors, evaluates their QUBO cost, and returns a deduplicated
:class:`~qubosolver.types.Solution` sorted by ascending cost.

It serves two roles in the broader solver stack:

* **Baseline solver** — used by :class:`~qubosolver.solvers.RandomSolver` as
  a standalone classical solver.
* **Warm-start initialiser** — used by
  :class:`~qubosolver.solvers.SimulatedAnnealingSolver` and
  :class:`~qubosolver.solvers.TabuSearchSolver` to generate a random starting
  point when no initial bitstring is provided.
"""

from __future__ import annotations

import torch

from qubosolver.types import Instance, Solution, bitstring, torch_rng
from qubosolver._utils import costs


def random_solutions(
    Q: Instance,
    *,
    max_bitstrings: int = 1,
    rng: torch.Generator = torch_rng(),
) -> Solution:
    """Sample uniformly random bitstring solutions for a QUBO instance.

    Draws *max_bitstrings* independent binary vectors uniformly at random,
    deduplicates them (identical samples are merged and their draw count is
    accumulated in ``counts``), evaluates the QUBO cost of each unique
    bitstring, and returns the result sorted by ascending cost with
    probabilities computed from the counts.

    .. note::
        The *rng* default is a :class:`torch.Generator` created **once** at
        module import time.  Pass an explicit generator when reproducibility
        across calls is required (e.g. ``torch_rng(seed=42)``).

    .. note::
        Because of deduplication, the returned solution may contain fewer than
        *max_bitstrings* bitstrings when the same random vector is drawn more
        than once.

    Args:
        Q: The QUBO instance whose coefficient matrix is used to evaluate
            bitstring costs.
        max_bitstrings: Number of random bitstrings to draw before
            deduplication.  The returned solution may contain fewer unique
            bitstrings.  Defaults to ``1``.
        rng: PyTorch random number generator controlling the sampling.
            Defaults to a module-level generator (see note above).

    Returns:
        A :class:`~qubosolver.types.Solution` with unique bitstrings,
        their QUBO costs, draw counts, and normalised probabilities, sorted
        by ascending cost.
    """
    bitstrings_ = bitstring.from_torch(
        torch.randint(0, 2, size=(max_bitstrings, Q.size), generator=rng)
    )
    unique_bits, counts = torch.unique(bitstrings_, dim=0, return_counts=True)
    costs_ = costs.batched_quadratic_cost(unique_bits.to(Q.matrix), Q.matrix)
    solution = (
        Solution(
            bitstrings=unique_bits,
            costs=costs_,
            counts=counts,
        )
        .sort_by_cost()
        .compute_probabilities()
    )

    return solution
