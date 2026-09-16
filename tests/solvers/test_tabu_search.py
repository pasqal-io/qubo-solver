from __future__ import annotations

import copy

import pytest
import pytest_check as check

import torch

from qubosolver import Instance, analysis, bitstrings, matrix, solving, vectori

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

    solution = solving.tabu_search.solve(instance, starts=start, max_iter=200)

    true_solution = copy.deepcopy(solution)._compute_costs(instance.matrix)

    torch.testing.assert_close(solution.costs, true_solution.costs)
    torch.testing.assert_close(
        solution.costs, torch.sort(solution.costs).values, atol=0.0, rtol=0.0
    )


def test_tabu_search_stays_consistent_across_refresh_boundary() -> None:
    """The incrementally tracked `QX`/`f_current` are periodically recomputed
    exactly (see `_REFRESH_EVERY` in `tabu_search.py`) to bound rounding
    drift; run past several refreshes and check the result is still fully
    internally consistent."""
    start = bitstrings.zeros(3, instance.size)

    solution = solving.tabu_search.solve(instance, starts=start, max_iter=300)

    check.is_true(solution.check_consistency(instance=instance, throw=True))


def test_tabu_search_runs_start_from_given_bitstrings() -> None:
    """Each row of ``start`` must seed its own independent run.

    Regression test for the bug where all parallel runs were seeded from a
    single repeated start rather than from the caller-provided batch: with
    one run per start and zero iterations, the result must equal the inputs
    (after dedup and cost sort), not a repetition of a single start.
    """
    start = bitstrings.from_strings(["000000", "111111", "101010", "000000"])

    solution = solving.tabu_search.solve(instance, starts=start, max_iter=0)

    expected_bitstrings = bitstrings.from_strings(["101010", "111111", "000000"])
    expected_counts = vectori.tensor([1, 1, 2])

    check.equal(len(solution), 3)
    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_tabu_search_is_deterministic_given_same_start() -> None:
    start = bitstrings.from_strings(["000000", "111111"])

    solution_a = solving.tabu_search.solve(instance, starts=start, max_iter=100)
    solution_b = solving.tabu_search.solve(instance, starts=start, max_iter=100)

    torch.testing.assert_close(solution_a.bitstrings, solution_b.bitstrings)
    torch.testing.assert_close(solution_a.costs, solution_b.costs)


def test_int_starts_generates_that_many_random_runs() -> None:
    """Passing an int for `starts` must generate that many uniformly random
    starting bitstrings, one independent run each."""
    n_starts = 5

    solution = solving.tabu_search.solve(instance, starts=n_starts, max_iter=50)

    check.is_true(solution.check_consistency(instance=instance, throw=True))
    check.less_equal(len(solution), n_starts)


def test_default_starts_is_one_random_start() -> None:
    """Omitting `starts` must default to a single uniformly random start."""
    solution = solving.tabu_search.solve(instance, max_iter=50)

    check.is_true(solution.check_consistency(instance=instance, throw=True))
    check.equal(len(solution), 1)


def test_tabu_search_never_lets_f_current_go_inf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`f_current` must always hold a real cost, never the `inf` sentinel.

    Invariant: `f_current` is the cost of the run's current `X`, so it is
    finite at every iteration -- and hence so is `f_candidates = f_current
    + dE` wherever a move is allowed, since `dE` is finite too.

    The sentinel can leak in because a run with no allowed move (every bit
    tabu, none meeting aspiration) has an all-`inf` masked row, whose
    minimum is that sentinel rather than a cost. `f_current` is persistent
    state, so assigning it there poisons every later iteration and also
    disables aspiration (`inf < f_best` is never true), leaving the run
    unable to search or to recover. Hence `f_current` is read from the
    unmasked candidate costs, not from the masked tensor.

    Checked by spying on the masking call: `f_candidates` must be finite
    wherever moves are allowed.

    Reaching an all-tabu row needs ``n <= tabu_tenure``, so a run can make
    every bit tabu before any tenure expires; any symmetric 2x2 does it.
    """
    n = 2
    small_instance = Instance(matrix=matrix.tensor([[-3.0, 1.0], [1.0, -3.0]]))
    starts = bitstrings.zeros(1, n)

    real_where = torch.where
    saw_all_tabu_iteration = False
    saw_inf_at_allowed_move = False

    def spying_where(condition: torch.Tensor, x: torch.Tensor, y: object) -> torch.Tensor:
        nonlocal saw_all_tabu_iteration, saw_inf_at_allowed_move
        if condition.any() and torch.isinf(x[condition]).any():
            saw_inf_at_allowed_move = True
        if not condition.any():
            saw_all_tabu_iteration = True
        return real_where(condition, x, y)

    monkeypatch.setattr(torch, "where", spying_where)

    solving.tabu_search.solve(
        small_instance,
        starts=starts,
        max_iter=50,
        tabu_tenure=7,
        max_no_improve=1000,
    )

    check.is_true(
        saw_all_tabu_iteration,
        "no all-tabu iteration was observed, so the invariant below went untested: "
        "either this spy no longer intercepts the masking call, or this instance no "
        "longer reaches an all-tabu row. Fix the test, not the assertion.",
    )
    check.is_false(
        saw_inf_at_allowed_move,
        "`f_candidates` was `inf` at an allowed move, so `f_current` is carrying the "
        "masking sentinel instead of a real cost",
    )


def test_tabu_search_still_finds_optimum_when_all_moves_become_tabu() -> None:
    """A run that hits an all-tabu row must keep searching afterwards.

    Requirement: encountering an iteration with no allowed move is normal
    and recoverable -- the run gives up that one move, not the search. On an
    instance small enough to solve exactly, it must still reach the optimum.

    This is the black-box counterpart of
    `test_tabu_search_never_lets_f_current_go_inf`: the same defect (the
    masking sentinel reaching persistent `f_current` state) seen through
    search quality rather than by spying on internals. Note that no
    assertion comparing a reported cost against its own bitstring can catch
    it -- a run carrying the sentinel can never beat `f_best`, so the best
    cost and bitstring stay mutually consistent and merely stop improving.
    Lost search progress is the only externally visible symptom.

    Two preconditions are load-bearing and neither is self-evident:

    - ``n <= tabu_tenure``, so a run can make every bit tabu before any
      tenure expires and an all-tabu row is actually reached.
    - `max_iter` well below ~500. A run carrying the sentinel is eventually
      rescued by an expiring tenure and does find the optimum, which would
      make this test pass for the wrong reason.

    `Q` was selected by search over random symmetric matrices -- only ~1 in
    120 makes the difference observable at this horizon -- so it is not
    interchangeable with an arbitrary matrix of the same size.
    """
    instance = Instance(matrix.tensor(
        [
            [-0.9875382781028748, 0.628231406211853, -0.10774004459381104, 1.2844055891036987],
            [0.628231406211853, -0.45010146498680115, 0.5505266189575195, -0.4758329391479492],
            [-0.10774004459381104, 0.5505266189575195, 0.4487120509147644, -0.22096946835517883],
            [1.2844055891036987, -0.4758329391479492, -0.22096946835517883, -0.5388421416282654],
        ]
    ))

    solution = solving.tabu_search.solve(
        instance,
        starts=bitstrings.zeros(1, instance.size),
        max_iter=100,
        tabu_tenure=7,
        max_no_improve=1000,
    )

    optimum = solving.brute_force.solve(instance)

    check.almost_equal(solution[0].cost, optimum[0].cost)

    print(f"\n{analysis.to_dataframe([optimum, solution], labels=["optimum", "solution"])}")
