from __future__ import annotations

import copy
import pytest_check as check

import torch

from qubosolver import Instance, bitstrings, matrix, solvers, vectori

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


def test_tabu_search_costs_match_bitstrings() -> None:
    """Every reported cost must correspond to x^T Q x of its own bitstring."""
    start = bitstrings.zeros(3, instance.size)

    solution = solvers.tabu_search(instance, start, max_iter=200)

    true_solution = copy.deepcopy(solution).compute_costs(instance.matrix)

    torch.testing.assert_close(solution.costs, true_solution.costs)
    torch.testing.assert_close(
        solution.costs, torch.sort(solution.costs).values, atol=0.0, rtol=0.0
    )


def test_tabu_search_runs_start_from_given_bitstrings() -> None:
    """Each row of ``start`` must seed its own independent run.

    Regression test for the bug where all parallel runs were seeded from a
    single repeated start rather than from the caller-provided batch: with
    one run per start and zero iterations, the result must equal the inputs
    (after dedup and cost sort), not a repetition of a single start.
    """
    start = bitstrings.from_strings(["000000", "111111", "101010", "000000"])

    solution = solvers.tabu_search(instance, start, max_iter=0)

    expected_bitstrings = bitstrings.from_strings(["101010", "111111", "000000"])
    expected_counts = vectori.tensor([1, 1, 2])

    check.equal(len(solution), 3)
    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_tabu_search_is_deterministic_given_same_start() -> None:
    start = bitstrings.from_strings(["000000", "111111"])

    solution_a = solvers.tabu_search(instance, start, max_iter=100)
    solution_b = solvers.tabu_search(instance, start, max_iter=100)

    torch.testing.assert_close(solution_a.bitstrings, solution_b.bitstrings)
    torch.testing.assert_close(solution_a.costs, solution_b.costs)
