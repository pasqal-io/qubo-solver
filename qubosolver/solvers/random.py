from __future__ import annotations

import torch

from qubosolver.types import QUBOInstance, QUBOSolution, bitstring, torch_rng
from qubosolver._utils import costs


def random_solutions(
    Q: QUBOInstance,
    *,
    max_bitstrings: int = 1,
    rng: torch.Generator = torch_rng(),
) -> QUBOSolution:
    """Generate random bitstring solutions for a QUBO instance.

    Samples ``max_bitstrings`` uniformly random binary vectors, deduplicates them,
    evaluates their costs, and returns the result sorted by ascending cost.

    Args:
        Q: The :class:`QUBOInstance` to evaluate against.
        rng: Random number generator for reproducibility.
        max_bitstrings: Number of random bitstrings to sample.

    Returns:
        A :class:`QUBOSolution` with unique bitstrings, costs, counts, and probabilities.
    """
    bitstrings_ = bitstring.from_torch(
        torch.randint(0, 2, size=(max_bitstrings, Q.size), generator=rng)
    )
    unique_bits, counts = torch.unique(bitstrings_, dim=0, return_counts=True)
    costs_ = costs.batched_quadratic_cost(unique_bits.to(Q.matrix), Q.matrix)
    solution = (
        QUBOSolution(
            bitstrings=unique_bits,
            costs=costs_,
            counts=counts,
        )
        .sort_by_cost()
        .compute_probabilities()
    )

    return solution
