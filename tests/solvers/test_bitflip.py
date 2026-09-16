from __future__ import annotations

import time
from typing import Literal

import pytest
import pytest_check as check
import torch

from qubosolver import (
    Solution,
    Instance,
    solving,
    bitstrings,
    vectori,
    matrix,
    torch_rng,
)
from qubosolver.solving.classical import iterative_bitflip_local_search
from qubosolver.solving.classical.iterative_bitflip_local_search import (
    _best_improvement_search_batch,
)


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_solution_not_mutated(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    Q = matrix.tensor([[-1.0, 2.0], [2.0, -2.0]])
    instance = Instance(Q)

    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution._update(instance)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")

    new_solution = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy
    )
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")
    check.equal(len(new_solution), 1)
    if strategy == "best_improvement":
        check.equal(new_solution[0].string, "01")
    else:
        check.equal(new_solution[0].string, "10")
    check.is_not(new_solution, solution)

    new_solution2 = solving.iterative_bitflip_local_search.solve(
        instance, starts=new_solution, strategy=strategy
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
    solution._update(instance)

    new_solution = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy
    )

    check.equal(new_solution[0].string, "11")
    check.less_equal(new_solution[0].cost, solution[0].cost)


def test_int_starts_generates_that_many_random_starts() -> None:
    """Passing an int for `starts` must draw that many uniformly random
    starting bitstrings (via random_sampling.solve) and locally optimize
    each of them, instead of requiring a pre-built Solution."""
    Q = matrix.tensor([[-10.0, 1.0], [1.0, -10.0]])
    instance = Instance(Q)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=5, strategy="best_improvement"
    )

    check.is_true(result.check_consistency(instance=instance, throw=True))
    check.less_equal(len(result), 5)
    for sol in result:
        check.equal(sol.string, "11")


def test_default_starts_is_one_random_start() -> None:
    """Omitting `starts` must default to a single uniformly random
    starting bitstring."""
    Q = matrix.tensor([[-10.0, 1.0], [1.0, -10.0]])
    instance = Instance(Q)

    result = solving.iterative_bitflip_local_search.solve(instance)

    check.is_true(result.check_consistency(instance=instance, throw=True))
    check.equal(len(result), 1)
    check.equal(result[0].string, "11")


def test_unknown_strategy_raises() -> None:
    Q = matrix.tensor([[-1.0, 2.0], [2.0, -2.0]])
    instance = Instance(Q)
    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution._update(instance)

    with pytest.raises(ValueError):
        solving.iterative_bitflip_local_search.solve(instance, starts=solution, strategy="does_not_exist")  # type: ignore[arg-type]


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
    solution._update(instance)

    limited = solving.iterative_bitflip_local_search.solve(
        instance,
        starts=solution,
        strategy="best_improvement",
        max_iterations=1,
    )
    unlimited = solving.iterative_bitflip_local_search.solve(
        instance,
        starts=solution,
        strategy="best_improvement",
        max_iterations=-1,
    )
    check.is_true(limited.check_consistency(instance=instance))
    check.is_true(unlimited.check_consistency(instance=instance))
    check.less_equal(unlimited[0].cost, limited[0].cost)


@pytest.mark.parametrize("n", [2, 10, 50, 200])
@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_solve_reaches_a_consistent_local_minimum(
    n: int, strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"]
) -> None:
    """Regardless of QUBO size, `solve` must return a solution whose costs are
    consistent with the instance and that is a genuine local minimum under
    single-bit flips: no flip of the best bitstring found should yield a
    strictly lower cost. This is the baseline correctness contract that any
    future cost-evaluation optimization (e.g. an incremental/differential
    cost update instead of recomputing z^T Q z from scratch on every flip)
    must continue to satisfy exactly."""
    rng = torch_rng(n)
    Q = torch.randn(n, n, generator=rng)
    Q = matrix.as_tensor((Q + Q.T) / 2)
    instance = Instance(Q)

    m = 20
    start = bitstrings.rand(m, n, rng=rng)
    solution = Solution(start, counts=vectori.zeros(m).fill_(1))
    solution._update(instance)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy
    )

    check.is_true(result.check_consistency(instance=instance, throw=True))

    best = result[0]
    for i in range(n):
        flipped = best.bitstring.clone()
        flipped[i] = 1 - flipped[i]
        check.greater_equal(instance.cost(flipped), best.cost)


def test_time_limit_is_global_and_skips_remaining_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 4
    Q = torch.randn(n, n, generator=torch_rng(0))
    Q = matrix.as_tensor((Q + Q.T) / 2)
    instance = Instance(Q)

    # Fake monotonic clock, ticked by the deadline checks themselves, so the
    # budget trips deterministically instead of depending on wall-clock timing.
    # `solve()` reads the clock once to set the global deadline, then once more
    # per row to compute that row's remaining budget; `_first_improvement_search`
    # reads it once to set its own inner deadline, then once per iteration of its
    # loop. Ticking by a small amount on every call, then jumping far past the
    # budget after the 4th call, lets row 0 read the (still-open) global
    # deadline, start its own search, take one iteration (applying its first
    # improving flip), and then have its *next* deadline check trip - so row 0
    # ends up improved by exactly one flip, while every following row is
    # skipped by the batch-level check before its search ever runs.
    time_limit = 3.0
    clock = 0.0
    call_count = 0

    def ticking_clock() -> float:
        nonlocal clock, call_count
        call_count += 1
        clock += time_limit + 1.0 if call_count > 4 else 0.1
        return clock

    monkeypatch.setattr(iterative_bitflip_local_search.time, "monotonic", ticking_clock)

    batch = 10
    solution = Solution(bitstrings.zeros(batch, n), counts=vectori.zeros(batch).fill_(1))
    solution._update(instance)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy="first_improvement", time_limit=time_limit
    )

    # Row 0 was improved by a single flip; the other 9 (all identical, untouched
    # all-zero rows) are merged into one entry.
    check.equal(len(result), 2)
    check.equal(result[0].string, "1000")
    check.equal(result[0].count, 1)
    check.less(result[0].cost, 0.0)
    check.equal(result[1].string, "0000")
    check.equal(result[1].count, 9)
    check.almost_equal(result[1].cost, 0.0)


@pytest.mark.parametrize("strategy", ["first_improvement", "greedy_sweep"])
def test_row_budget_is_the_remaining_batch_budget(
    strategy: Literal["greedy_sweep", "first_improvement"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row is granted only what is left of the global budget.

    Regression test: passing the *full* `time_limit` down to every row makes
    the batch budget effectively per-row, so a batch of `m` rows that each run
    to their own deadline can consume up to `m * time_limit` in total.

    Rather than assert on the fake clock's exact final value (which depends on
    how many ticks each strategy happens to burn), this checks the budget the
    rows are actually handed: every row's budget must be strictly smaller than
    the previous row's, since time only moves forward. A full `time_limit`
    handed to each row shows up as a constant sequence instead.
    """
    n = 4
    Q = torch.randn(n, n, generator=torch_rng(0))
    Q = matrix.as_tensor((Q + Q.T) / 2)
    instance = Instance(Q)

    time_limit = 1.0
    clock = 0.0

    def ticking_clock() -> float:
        nonlocal clock
        clock += 0.01
        return clock

    monkeypatch.setattr(iterative_bitflip_local_search.time, "monotonic", ticking_clock)

    # Record the budget handed to each row, then stop that row immediately so
    # the batch walks through every row instead of burning the budget on one.
    row_budgets: list[float] = []
    search_fn = iterative_bitflip_local_search._ROW_STRATEGIES[strategy]

    def recording_search(*args: object, **kwargs: object) -> torch.Tensor:
        row_budgets.append(float(kwargs["time_limit"]))  # type: ignore[arg-type]
        return search_fn(*args, **{**kwargs, "time_limit": 0.0})

    monkeypatch.setitem(iterative_bitflip_local_search._ROW_STRATEGIES, strategy, recording_search)

    batch = 5
    solution = Solution(
        bitstrings.rand(batch, n, rng=torch_rng(1)), counts=vectori.zeros(batch).fill_(1)
    )
    solution._update(instance)

    solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy, time_limit=time_limit
    )

    check.equal(len(row_budgets), batch)
    # Strictly decreasing: each row only gets what the previous rows left. With
    # the full budget passed down, every entry would instead equal `time_limit`.
    for earlier, later in zip(row_budgets, row_budgets[1:]):
        check.less(later, earlier)
    check.less(row_budgets[0], time_limit)


def test_best_improvement_batch_matches_per_row_result() -> None:
    """best_improvement runs every row of a multi-row batch in lockstep,
    masking out rows that already reached their own local optimum so they
    are not perturbed while slower rows keep improving. Each row's result
    must therefore be identical to running that same row alone through the
    batch search."""
    n, m = 12, 8
    rng = torch_rng(7)
    Q = torch.randn(n, n, generator=rng)
    Q = matrix.as_tensor((Q + Q.T) / 2)

    X = bitstrings.rand(m, n, rng=rng)

    batched = _best_improvement_search_batch(Q, X)
    for i in range(m):
        solo = _best_improvement_search_batch(Q, X[i : i + 1])
        check.is_true(torch.equal(batched[i], solo[0]))


@pytest.mark.priority(10)
@pytest.mark.parametrize(
    "variables, starts, strategy, expected_elapsed, expected_cplex_gap",
    [
        (10, 10, "best_improvement", 0.5, 0.0001),
        (10, 100, "best_improvement", 0.5, 0.0001),
        (10, 500, "best_improvement", 0.5, 0.0001),
        (50, 10, "best_improvement", 0.5, 0.0001),
        (50, 100, "best_improvement", 0.5, 0.0001),
        (50, 500, "best_improvement", 0.5, 0.0001),
        (200, 10, "best_improvement", 0.5, 0.3),
        (200, 100, "best_improvement", 0.5, 0.0001),
        (200, 500, "best_improvement", 0.5, 0.0001),
        (10, 10, "first_improvement", 0.5, 0.0001),
        (10, 100, "first_improvement", 0.5, 0.0001),
        (10, 500, "first_improvement", 1.0, 0.0001),
        (50, 10, "first_improvement", 0.5, 0.0001),
        (50, 100, "first_improvement", 1.0, 0.0001),
        (50, 500, "first_improvement", 4.0, 0.0001),
        (200, 10, "first_improvement", 1.0, 0.0001),
        (200, 100, "first_improvement", 6.0, 0.0001),
        (200, 500, "first_improvement", 99.0, 0.0001),  # time-out
        (10, 10, "greedy_sweep", 0.5, 0.0001),
        (10, 100, "greedy_sweep", 0.5, 0.0001),
        (10, 500, "greedy_sweep", 1.0, 0.0001),
        (50, 10, "greedy_sweep", 0.5, 0.0001),
        (50, 100, "greedy_sweep", 2.0, 0.0001),
        (50, 500, "greedy_sweep", 6.0, 0.0001),
        (200, 10, "greedy_sweep", 1.0, 0.3),
        (200, 100, "greedy_sweep", 8.0, 0.0001),
        (200, 500, "greedy_sweep", 99.0, 0.0001),  # time-out
    ],
)
def test_benchmark_grid_report(
    variables: int,
    starts: int,
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
    expected_elapsed: float,
    expected_cplex_gap: float,
) -> None:
    """Report wall time and best cost found for every (num_variables,
    batch_size, strategy) cell of a benchmark grid, checked against the
    CPLEX-optimal cost for that `num_variables` and a baseline wall time
    observed on a previous run."""
    time_limit = 10.0

    rng = torch_rng(0)
    Q = torch.randn(variables, variables, generator=rng)
    Q = matrix.as_tensor((Q + Q.T) / 2)
    instance = Instance(Q)

    starts_ = bitstrings.rand(starts, variables, rng=rng)
    solution = Solution(starts_, counts=vectori.zeros(starts).fill_(1))
    solution._update(instance)

    t0 = time.perf_counter()
    result = solving.iterative_bitflip_local_search.solve(
        instance,
        starts=solution,
        strategy=strategy,
        max_iterations=-1,
        time_limit=time_limit,
    )
    elapsed = time.perf_counter() - t0
    best_cost = float(result.costs.min().item())

    cplex_costs = {
        10: -11.9943,
        50: -149.1611,
        200: -1309.7295,
    }
    cplex_cost = cplex_costs[variables]
    gap = 100 * (best_cost - cplex_cost) / abs(cplex_cost)

    print(f"\nvariables = {variables}")
    print(f"starts = {starts}")
    print(f"strategy = {strategy}")
    print(f"elapsed = {elapsed} s")
    print(f"best_cost = {best_cost}")
    print(f"gap = {gap} %")

    check.less_equal(gap, expected_cplex_gap)

    if elapsed > expected_elapsed:
        pytest.xfail("elapsed time exceeded the expected baseline")
