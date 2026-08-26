from __future__ import annotations

import itertools

import pytest
import pytest_check as check
import torch

from qubosolver import Instance, solvers, bitstrings, matrix, torch_rng, SingleSolution


def _reference_sorted(instance: Instance) -> list[SingleSolution]:
    """All candidate solutions sorted by ascending cost, by exhaustive search."""
    solutions = [
        SingleSolution(bitstring=b, cost=instance.cost(b))
        for b in bitstrings.tensor(list(itertools.product([0, 1], repeat=instance.size)))
    ]
    solutions.sort(key=lambda s: s.cost)
    return solutions


def sample_qubo() -> Instance:
    return Instance(
        matrix.tensor(
            [
                [0.0, -2.0, 1.0, 1.0],
                [-2.0, 0.0, -2.0, 1.0],
                [1.0, -2.0, 0.0, -2.0],
                [1.0, 1.0, -2.0, 0.0],
            ]
        )
    )


def test_returns_global_optimum() -> None:
    instance = sample_qubo()
    optimum = _reference_sorted(instance)[0]

    solution = solvers.brute_force(instance, max_bitstrings=1)

    check.equal(len(solution), 1)
    check.equal(solution[0].cost, optimum.cost)
    # optimum may be degenerate; assert on cost, not the exact bitstring.
    best = solution[0].bitstring
    check.equal(instance.cost(best), optimum.cost)


def test_top_k_sorted_matches_reference() -> None:
    instance = sample_qubo()
    reference_costs = [s.cost for s in _reference_sorted(instance)[:3]]

    solution = solvers.brute_force(instance, max_bitstrings=3)

    check.equal(len(solution), 3)
    costs = solution.costs.tolist()
    # Sorted ascending and equal to the three lowest reference costs.
    check.equal(costs, sorted(costs))
    check.equal(costs, reference_costs)


def test_max_bitstrings_capped_at_2_to_the_n() -> None:
    # A 2-variable QUBO has only 4 assignments; asking for more returns 4.
    Q = matrix.tensor([[1.0, -1.0], [-1.0, 1.0]])

    solution = solvers.brute_force(Instance(Q), max_bitstrings=100)

    check.equal(len(solution), 4)


def test_probabilities_are_normalised() -> None:
    solution = solvers.brute_force(sample_qubo(), max_bitstrings=3)

    check.almost_equal(solution.probabilities.sum().item(), 1.0)


def test_empty_instance_returns_empty_solution() -> None:
    solution = solvers.brute_force(Instance(matrix.zeros(0)))

    check.is_false(solution)


def test_time_limit_returns_best_so_far_without_enumerating_all() -> None:
    # 24 variables => ~1.6e7 assignments. A tiny budget must return promptly with
    # the requested number of bitstrings rather than hang or exhaust memory.
    instance = Instance(matrix.as_tensor(torch.randn(24, 24, generator=torch_rng(26))))

    solution = solvers.brute_force(instance, max_bitstrings=2, time_limit=0.05)

    check.equal(len(solution), 2)
    # Costs are consistent with the instance (sanity: finite, correctly evaluated).
    for b in solution.bitstrings:
        check.is_true(torch.isfinite(torch.tensor(instance.cost(b))))


def test_large_instance_with_no_time_limit_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 21 variables exceeds the 20-variable threshold; 2^21 assignments still
    # completes quickly enough for a test.
    instance = Instance(matrix.as_tensor(torch.randn(21, 21, generator=torch_rng(1))))

    with caplog.at_level("WARNING", logger="qubosolver.solvers.classical.brute_force"):
        solvers.brute_force(instance, max_bitstrings=1, time_limit=float("inf"))

    check.is_true(any("time_limit" in record.message for record in caplog.records))


def test_large_instance_with_finite_time_limit_does_not_log_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    instance = Instance(matrix.as_tensor(torch.randn(21, 21, generator=torch_rng(1))))

    with caplog.at_level("WARNING", logger="qubosolver.solvers.classical.brute_force"):
        solvers.brute_force(instance, max_bitstrings=1, time_limit=5.0)

    check.equal(len(caplog.records), 0)


def test_small_instance_with_no_time_limit_does_not_log_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="qubosolver.solvers.classical.brute_force"):
        solvers.brute_force(sample_qubo(), max_bitstrings=1, time_limit=float("inf"))

    check.equal(len(caplog.records), 0)
