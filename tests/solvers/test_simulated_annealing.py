from __future__ import annotations

import torch
import copy

from qubosolver import Instance, bitstring, torch_rng, solvers, matrix

instance = Instance(
    matrix.tensor(
        [
            [-2.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, -3.0, 2.0, 0.0, 0.0, -1.0],
            [0.0, 2.0, -1.0, 1.0, -2.0, 0.0],
            [-1.0, 0.0, 1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, -2.0, 1.0, -1.0, 2.0],
            [0.0, -1.0, 0.0, 0.0, 2.0, -2.0],
        ],
    )
)


def test_simulated_annealing_costs_match_bitstrings() -> None:
    """Every reported cost must correspond to x^T Q x of its own bitstring."""
    start = bitstring.zeros(instance.size)
    rng = torch_rng(0)

    solution = solvers.simulated_annealing(
        instance,
        start,
        top_k=5,
        max_iter=3000,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
    )

    true_solution = copy.deepcopy(solution).compute_costs(instance.matrix)

    torch.testing.assert_close(solution.costs, true_solution.costs)
    torch.testing.assert_close(
        solution.costs, torch.sort(solution.costs).values, atol=0.0, rtol=0.0
    )
