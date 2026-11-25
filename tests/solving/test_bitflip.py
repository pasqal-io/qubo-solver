# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import pytest_check as check

from qubosolver import Instance, Solution, bitstrings, numpy_rng, solving
from qubosolver.solving.classical import iterative_bitflip_local_search
from qubosolver.solving.classical.iterative_bitflip_local_search import (
    _best_improvement_search_batch,
    _first_improvement_search,
    _greedy_sweep_search,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_solution_not_mutated(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    Q = np.asarray([[-1.0, 2.0], [2.0, -2.0]], dtype=np.float64)
    instance = Instance(Q)

    solution = Solution(np.zeros((1, 2), dtype=np.int8), counts=np.asarray([1], dtype=np.int64))
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
    Q = np.asarray([[-10.0, 1.0], [1.0, -10.0]], dtype=np.float64)
    instance = Instance(Q)

    solution = Solution(np.zeros((1, 2), dtype=np.int8), counts=np.asarray([1], dtype=np.int64))
    solution._update(instance)

    new_solution = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy
    )

    check.equal(new_solution[0].string, "11")
    check.less_equal(new_solution[0].cost, solution[0].cost)


def test_int_starts_generates_that_many_random_starts() -> None:
    """Passing an int for `starts` must draw that many uniformly random starting bitstrings.

    They are drawn via random_sampling.solve and each is locally optimized, instead of
    requiring a pre-built Solution.
    """
    Q = np.asarray([[-10.0, 1.0], [1.0, -10.0]], dtype=np.float64)
    instance = Instance(Q)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=5, strategy="best_improvement"
    )

    check.is_true(result.check_consistency(instance=instance, throw=True))
    check.less_equal(len(result), 5)
    for sol in result:
        check.equal(sol.string, "11")


def test_default_starts_is_one_random_start() -> None:
    """Omitting `starts` must default to a single uniformly random starting bitstring."""
    Q = np.asarray([[-10.0, 1.0], [1.0, -10.0]], dtype=np.float64)
    instance = Instance(Q)

    result = solving.iterative_bitflip_local_search.solve(instance)

    check.is_true(result.check_consistency(instance=instance, throw=True))
    check.equal(len(result), 1)
    check.equal(result[0].string, "11")


def test_unknown_strategy_raises() -> None:
    Q = np.asarray([[-1.0, 2.0], [2.0, -2.0]], dtype=np.float64)
    instance = Instance(Q)
    solution = Solution(np.zeros((1, 2), dtype=np.int8), counts=np.asarray([1], dtype=np.int64))
    solution._update(instance)

    with pytest.raises(ValueError):
        solving.iterative_bitflip_local_search.solve(
            instance,
            starts=solution,
            strategy="does_not_exist",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement", "greedy_sweep"])
def test_starts_length_mismatch_raises(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    """`starts` bitstrings shorter or longer than `instance.size` must raise, not crash."""
    Q = np.asarray([[-1.0, 2.0], [2.0, -2.0]], dtype=np.float64)
    instance = Instance(Q)

    short_solution = Solution(
        np.zeros((1, 1), dtype=np.int8), counts=np.asarray([1], dtype=np.int64)
    )
    long_solution = Solution(
        np.zeros((1, 3), dtype=np.int8), counts=np.asarray([1], dtype=np.int64)
    )

    with pytest.raises(ValueError, match=r"instance\.size"):
        solving.iterative_bitflip_local_search.solve(
            instance, starts=short_solution, strategy=strategy
        )

    with pytest.raises(ValueError, match=r"instance\.size"):
        solving.iterative_bitflip_local_search.solve(
            instance, starts=long_solution, strategy=strategy
        )


@pytest.mark.parametrize("strategy", ["best_improvement", "first_improvement"])
def test_max_iterations_limits_progress(
    strategy: Literal["best_improvement", "first_improvement"],
) -> None:
    # Chained improvements: flipping bit 0 helps, then flipping bit 1 helps further.
    Q = np.asarray(
        [
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 0.0],
            [5.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    instance = Instance(Q)
    solution = Solution(np.zeros((1, 3), dtype=np.int8), counts=np.asarray([1], dtype=np.int64))
    solution._update(instance)

    limited = solving.iterative_bitflip_local_search.solve(
        instance,
        starts=solution,
        strategy=strategy,
        max_iterations=1,
    )
    unlimited = solving.iterative_bitflip_local_search.solve(
        instance,
        starts=solution,
        strategy=strategy,
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
    """Regardless of QUBO size, `solve` must return a consistent, genuine local minimum.

    The solution's cost must be consistent with the instance, and it must be a genuine
    local minimum under single-bit flips: no flip of the best bitstring found should yield
    a strictly lower cost. This is the baseline correctness contract that any future
    cost-evaluation optimization (e.g. an incremental/differential cost update instead of
    recomputing z^T Q z from scratch on every flip) must continue to satisfy exactly.
    """
    rng = numpy_rng(n)
    Q = rng.standard_normal((n, n))
    Q = np.asarray((Q + Q.T) / 2, dtype=np.float64)
    instance = Instance(Q)

    m = 20
    start = bitstrings.rand(m, n, rng=rng)
    solution = Solution(start, counts=np.zeros(m, dtype=np.int64) + 1)
    solution._update(instance)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy
    )

    check.is_true(result.check_consistency(instance=instance, throw=True))

    best = result[0]
    for i in range(n):
        flipped = best.bitstring.copy()
        flipped[i] = 1 - flipped[i]
        check.greater_equal(instance.cost(flipped), best.cost)


def test_time_limit_is_global_and_skips_remaining_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 4
    Q = numpy_rng(0).standard_normal((n, n))
    Q = np.asarray((Q + Q.T) / 2, dtype=np.float64)
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
    solution = Solution(
        np.zeros((batch, n), dtype=np.int8), counts=np.zeros(batch, dtype=np.int64) + 1
    )
    solution._update(instance)

    result = solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy="first_improvement", time_limit=time_limit
    )

    # Row 0 was improved by a single flip; the other 9 (all identical, untouched
    # all-zero rows) are merged into one entry.
    check.equal(len(result), 2)
    check.equal(result[0].string, "0010")
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
    Q = numpy_rng(0).standard_normal((n, n))
    Q = np.asarray((Q + Q.T) / 2, dtype=np.float64)
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

    def recording_search(*args: object, **kwargs: object) -> np.ndarray:
        row_budgets.append(float(kwargs["time_limit"]))  # type: ignore[arg-type]
        return search_fn(*args, **{**kwargs, "time_limit": 0.0})

    monkeypatch.setitem(iterative_bitflip_local_search._ROW_STRATEGIES, strategy, recording_search)

    batch = 5
    solution = Solution(
        bitstrings.rand(batch, n, rng=numpy_rng(1)), counts=np.zeros(batch, dtype=np.int64) + 1
    )
    solution._update(instance)

    solving.iterative_bitflip_local_search.solve(
        instance, starts=solution, strategy=strategy, time_limit=time_limit
    )

    check.equal(len(row_budgets), batch)
    # Strictly decreasing: each row only gets what the previous rows left. With
    # the full budget passed down, every entry would instead equal `time_limit`.
    for earlier, later in itertools.pairwise(row_budgets):
        check.less(later, earlier)
    check.less(row_budgets[0], time_limit)


def test_best_improvement_batch_matches_per_row_result() -> None:
    """best_improvement runs every row of a multi-row batch in lockstep.

    It masks out rows that already reached their own local optimum so they are not
    perturbed while slower rows keep improving. Each row's result must therefore be
    identical to running that same row alone through the batch search.
    """
    n, m = 12, 8
    rng = numpy_rng(7)
    Q = rng.standard_normal((n, n))
    Q = np.asarray((Q + Q.T) / 2, dtype=np.float64)

    X = bitstrings.rand(m, n, rng=rng)

    batched = _best_improvement_search_batch(Q, X)
    for i in range(m):
        solo = _best_improvement_search_batch(Q, X[i : i + 1])
        check.is_true(np.array_equal(batched[i], solo[0]))


@pytest.mark.parametrize(
    "variables, starts, strategy, expected_elapsed, expected_cplex_gap",
    [
        (10, 10, "best_improvement", 0.5, 0.0001),
        (10, 100, "best_improvement", 0.5, 0.0001),
        (10, 500, "best_improvement", 0.5, 0.0001),
        (50, 10, "best_improvement", 0.5, 0.0001),
        (50, 100, "best_improvement", 0.5, 0.0001),
        (50, 500, "best_improvement", 0.5, 0.0001),
        (200, 10, "best_improvement", 0.5, 1.0),
        (200, 100, "best_improvement", 0.5, 0.2),
        (200, 500, "best_improvement", 0.5, 0.0001),
        (10, 10, "first_improvement", 0.5, 7.0),
        (10, 100, "first_improvement", 0.5, 0.0001),
        (10, 500, "first_improvement", 1.0, 0.0001),
        (50, 10, "first_improvement", 0.5, 0.0001),
        (50, 100, "first_improvement", 1.0, 0.0001),
        (50, 500, "first_improvement", 4.0, 0.0001),
        (200, 10, "first_improvement", 1.0, 1.0),
        (200, 100, "first_improvement", 6.0, 0.1),
        (200, 500, "first_improvement", 99.0, 0.0001),  # time-out
        (10, 10, "greedy_sweep", 0.5, 0.0001),
        (10, 100, "greedy_sweep", 0.5, 0.0001),
        (10, 500, "greedy_sweep", 1.0, 0.0001),
        (50, 10, "greedy_sweep", 0.5, 0.0001),
        (50, 100, "greedy_sweep", 2.0, 0.0001),
        (50, 500, "greedy_sweep", 6.0, 0.0001),
        (200, 10, "greedy_sweep", 1.0, 0.6),
        (200, 100, "greedy_sweep", 8.0, 0.3),
        (200, 500, "greedy_sweep", 99.0, 0.3),  # time-out
    ],
)
def test_benchmark_grid_report(
    variables: int,
    starts: int,
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
    expected_elapsed: float,
    expected_cplex_gap: float,
) -> None:
    """Report wall time and best cost found for every cell of a benchmark grid.

    Each (num_variables, batch_size, strategy) cell is checked against the CPLEX-optimal
    cost for that `num_variables` and a baseline wall time observed on a previous run.
    """
    time_limit = 10.0

    rng = numpy_rng(0)
    Q = rng.standard_normal((variables, variables))
    Q = np.asarray((Q + Q.T) / 2, dtype=np.float64)
    instance = Instance(Q)

    starts_ = bitstrings.rand(starts, variables, rng=rng)
    solution = Solution(starts_, counts=np.zeros(starts, dtype=np.int64) + 1)
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
        10: -5.8005,
        50: -166.1279,
        200: -1171.9005,  # best incumbent after 300 s, not proven optimal
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

    if elapsed > expected_elapsed:  # no cov
        pytest.xfail("elapsed time exceeded the expected baseline")


def test_search_functions_raise_on_zero_size_matrix() -> None:
    Q = np.zeros((0, 0), dtype=np.float64)
    X = np.zeros((1, 0), dtype=np.int8)

    with pytest.raises(ValueError):
        _best_improvement_search_batch(Q, X)

    with pytest.raises(ValueError):
        _first_improvement_search(Q, X[0])

    with pytest.raises(ValueError):
        _greedy_sweep_search(Q, X[0])


@pytest.mark.parametrize("strategy", ["greedy_sweep", "best_improvement", "first_improvement"])
def test_iterative_bitflip_local_search_solve_empty_instance(
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"],
) -> None:
    instance = Instance()

    solution = solving.iterative_bitflip_local_search.solve(instance, strategy=strategy)

    assert isinstance(solution, Solution)
    check.is_true(solution.check_consistency(instance=instance, throw=True))
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "")
    check.equal(solution[0].cost, 0.0)
    check.equal(solution[0].count, 1)
    check.equal(solution[0].probability, 1.0)


def test_no_solution() -> None:
    # fmt: off
    Q = np.asarray([
        [-10.0, 1.0],
        [1.0, -10.0]
    ], dtype=np.float64)
    # fmt: on

    instance = Instance(matrix=Q)
    solution = Solution()
    check.equal(len(solution), 0)

    # Bitflip doesn't find new solutions if there were none to begin with
    pp_solution = solving.iterative_bitflip_local_search.solve(instance, starts=solution)
    check.equal(len(pp_solution), 0)


@pytest.mark.parametrize(
    "search", [_first_improvement_search, _greedy_sweep_search, _best_improvement_search_batch]
)
def test_local_search_stops_at_the_deadline(search: Callable[..., np.ndarray]) -> None:
    # fmt: off
    Q = np.asarray([
        [-10.0, 1.0],
        [1.0, -10.0]
    ], dtype=np.float64)
    # fmt: on
    s = np.zeros(2, dtype=np.int8)
    if search is _best_improvement_search_batch:
        s = s[np.newaxis]
    # An already expired deadline stops the search before any flip.
    np.testing.assert_allclose(search(Q, s, time_limit=-1.0), s, rtol=0, strict=True)


@pytest.mark.parametrize("search", [_first_improvement_search, _greedy_sweep_search])
def test_local_search_with_rng(search: Callable[..., np.ndarray]) -> None:
    # fmt: off
    Q = np.asarray([
        [-10.0, 1.0],
        [1.0, -10.0]
    ], dtype=np.float64)
    # fmt: on
    best_bitstring = search(Q, np.zeros(2, dtype=np.int8), rng=numpy_rng(2005221736))
    np.testing.assert_allclose(
        best_bitstring, np.asarray([1, 1], dtype=np.int8), rtol=0, strict=True
    )


def test_greedy_sweep_stops_after_max_iterations_while_still_improving() -> None:
    # From 00, the first sweep only flips bit 1 (flipping bit 0 first would raise the
    # cost); a second sweep would then flip bit 0 and reach the optimum 11.
    Q = np.asarray([[1.0, -2.0], [-2.0, -0.5]], dtype=np.float64)
    start = np.zeros(2, dtype=np.int8)

    one_sweep = _greedy_sweep_search(Q, start, max_iterations=1)
    converged = _greedy_sweep_search(Q, start)

    np.testing.assert_allclose(one_sweep, np.asarray([0, 1], dtype=np.int8), rtol=0, strict=True)
    np.testing.assert_allclose(converged, np.asarray([1, 1], dtype=np.int8), rtol=0, strict=True)
