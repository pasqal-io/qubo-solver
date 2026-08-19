from __future__ import annotations

from typing import Literal

import pytest
import pytest_check as check
import torch

from qubosolver import (
    Solution,
    Instance,
    solvers,
    bitstrings,
    vectori,
    matrix,
    torch_rng,
    Bitstring,
)
from qubosolver.solvers.classical import bitflip


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_solution_not_mutated(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    Q = matrix.tensor([[-1.0, 2.0], [2.0, -2.0]])
    instance = Instance(Q)

    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution._compute_costs(instance.matrix)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")

    new_solution = solvers.iterative_bitflip_local_search(instance, solution, strategy=strategy)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")
    check.equal(len(new_solution), 1)
    if strategy == "best_improvement":
        check.equal(new_solution[0].string, "01")
    else:
        check.equal(new_solution[0].string, "10")
    check.is_not(new_solution, solution)

    new_solution2 = solvers.iterative_bitflip_local_search(
        instance, new_solution, strategy=strategy
    )
    check.equal(len(new_solution2), 1)
    if strategy == "best_improvement":
        check.equal(new_solution2[0].string, "01")
    else:
        check.equal(new_solution2[0].string, "10")


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_strategy_selection_improves_solution(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    Q = matrix.tensor([[-10.0, 1.0], [1.0, -10.0]])
    instance = Instance(Q)

    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution.compute_costs(instance.matrix)

    new_solution = solvers.iterative_bitflip_local_search(instance, solution, strategy=strategy)

    check.equal(new_solution[0].string, "11")
    check.less_equal(new_solution[0].cost, solution[0].cost)


def test_unknown_strategy_raises() -> None:
    Q = matrix.tensor([[-1.0, 2.0], [2.0, -2.0]])
    instance = Instance(Q)
    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution.compute_costs(instance.matrix)

    with pytest.raises(ValueError):
        solvers.iterative_bitflip_local_search(instance, solution, strategy="does_not_exist")  # type: ignore[arg-type]


def test_max_iterations_limits_progress() -> None:
    # Chained improvements: flipping bit 0 helps, then flipping bit 1 helps further.
    Q = matrix.tensor(
        [
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 0.0],
            [5.0, 0.0, -1.0],
        ]
    )
    instance = Instance(Q)
    solution = Solution(bitstrings.zeros(1, 3), counts=vectori.tensor([1]))
    solution.compute_costs(instance.matrix)

    limited = solvers.iterative_bitflip_local_search(
        instance,
        solution,
        strategy="best_improvement",
        max_iterations=1,
    )
    unlimited = solvers.iterative_bitflip_local_search(
        instance,
        solution,
        strategy="best_improvement",
        max_iterations=-1,
    )
    check.is_true(limited.check_consistency(instance))
    check.is_true(unlimited.check_consistency(instance))
    check.less_equal(unlimited[0].cost, limited[0].cost)


def test_time_limit_is_global_and_skips_remaining_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 4
    Q = torch.randn(n, n, generator=torch_rng(0))
    Q = matrix.tensor((Q + Q.T) / 2)
    instance = Instance(Q)

    # Fake monotonic clock, ticked by qubo evaluations, so the deadline trips
    # deterministically instead of depending on wall-clock timing.
    time_limit = 3.0
    clock = 0.0
    monkeypatch.setattr(bitflip.time, "monotonic", lambda: clock)

    batch = 10
    solution = Solution(bitstrings.zeros(batch, n), counts=vectori.zeros(batch).fill_(1))
    solution.compute_costs(Q)

    original_eval = instance.evaluate_solution
    eval_count = 0

    def ticking_eval(s: Bitstring) -> float:
        # Advance the clock by more than the whole budget on every evaluation. The
        # first row's search still gets to try its first flip (its own inner deadline
        # is computed only *after* the initial evaluation has already ticked the
        # clock forward), finds an improving flip on the first try, and applies it -
        # but by then the global deadline (fixed before the batch loop started) is
        # long past, so every subsequent row is skipped by the batch-level check
        # before it ever reaches evaluate_solution.
        nonlocal clock, eval_count
        eval_count += 1
        clock += time_limit + 1.0
        return original_eval(s)

    monkeypatch.setattr(instance, "evaluate_solution", ticking_eval)

    result = solvers.iterative_bitflip_local_search(
        instance, solution, strategy="first_improvement", time_limit=time_limit
    )

    # Only row 0 is ever searched: one eval for its initial cost, one more for the
    # first improving flip it applies before its own deadline check trips. Every
    # other row is skipped by the batch-level deadline check before being searched,
    # so evaluate_solution is called exactly twice for the whole batch.
    check.equal(eval_count, 2)

    # Row 0 was improved by a single flip; the other 9 (all identical, untouched
    # all-zero rows) are merged by the final torch.unique into one entry.
    check.equal(len(result), 2)
    check.equal(result[0].string, "1000")
    check.equal(result[0].count, 1)
    check.less(result[0].cost, 0.0)
    check.equal(result[1].string, "0000")
    check.equal(result[1].count, 9)
    check.almost_equal(result[1].cost, 0.0)
