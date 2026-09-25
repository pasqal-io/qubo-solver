from __future__ import annotations

import copy
import inspect
import sys

import pytest
import pytest_check as check
import torch
from typing_extensions import assert_type, get_overloads

from qubosolver import (
    Dataset,
    Instance,
    Solution,
    bitstring,
    bitstrings,
    matrix,
    solving,
    torch_rng,
    vectori,
)
from qubosolver.solving.classical.simulated_annealing import (
    _Data,
    _from_key,
    _item_energy,
    _shrink,
    _to_key,
)

instance_symmetric = Instance(
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

instance_small = Instance(
    matrix.tensor(
        [
            [1.0, 3.0, -1.0, 0.5],
            [3.0, -2.0, 1.0, 0.0],
            [-1.0, 1.0, 0.5, -1.5],
            [0.5, 0.0, -1.5, -1.0],
        ],
    )
)

instances = [instance_symmetric, instance_small]
instance_ids = ["6var", "4var"]

vectorized_params = [
    pytest.param(True, id="vectorized"),
    pytest.param(False, id="sequential"),
]

# The non-vectorized path does one Python-level torch call per (start, iteration)
# pair, so its wall-clock cost scales with n_starts * max_iter regardless of
# instance size; skip it once that product gets large enough to slow the suite.
_MAX_SEQUENTIAL_WORK = 2000


def skip_if_sequential_too_slow(vectorized: bool, *, n_starts: int, max_iter: int) -> None:
    if not vectorized and n_starts * max_iter > _MAX_SEQUENTIAL_WORK:
        pytest.skip("n_starts * max_iter too large for the non-vectorized path")


def test_to_key_from_key_round_trip() -> None:
    """_from_key must reconstruct the exact bitstring given to _to_key."""
    bits = bitstring.from_string("10110")

    key = _to_key(bits)
    restored = _from_key(key)

    torch.testing.assert_close(restored, bits)


def test_to_key_distinguishes_different_bitstrings() -> None:
    """Different bitstrings must map to different keys."""
    a = _to_key(bitstring.from_string("100"))
    b = _to_key(bitstring.from_string("001"))

    check.not_equal(a, b)


def test_to_key_equal_for_equal_bitstrings() -> None:
    """Two tensors with the same values must map to the same key."""
    a = _to_key(bitstring.from_string("101"))
    b = _to_key(bitstring.from_string("101"))

    check.equal(a, b)


def test_item_energy_reads_data_energy() -> None:
    """_item_energy must extract the energy field of the (key, _Data) pair."""
    item = (b"\x00", _Data(energy=-3.5, count=2))

    check.equal(_item_energy(item), -3.5)


def test_shrink_keeps_top_k_lowest_energy() -> None:
    """_shrink must retain only the top_k lowest-energy entries."""
    visited_solutions = {
        b"a": _Data(energy=3.0, count=1),
        b"b": _Data(energy=1.0, count=1),
        b"c": _Data(energy=2.0, count=1),
        b"d": _Data(energy=0.5, count=1),
    }

    _shrink(visited_solutions, top_k=2)

    check.equal(set(visited_solutions.keys()), {b"d", b"b"})


def test_shrink_mutates_in_place() -> None:
    """_shrink must mutate the passed-in dict rather than returning a new one."""
    visited_solutions = {
        b"a": _Data(energy=3.0, count=1),
        b"b": _Data(energy=1.0, count=1),
    }
    original = visited_solutions

    _shrink(visited_solutions, top_k=1)

    check.is_(visited_solutions, original)
    check.equal(len(visited_solutions), 1)


def test_shrink_noop_when_already_within_top_k() -> None:
    """_shrink must not drop entries when the dict already fits within top_k."""
    visited_solutions = {
        b"a": _Data(energy=3.0, count=1),
        b"b": _Data(energy=1.0, count=1),
    }

    _shrink(visited_solutions, top_k=5)

    check.equal(set(visited_solutions.keys()), {b"a", b"b"})


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_costs_match_bitstrings(instance: Instance, vectorized: bool) -> None:
    """Every reported cost must correspond to x^T Q x of its own bitstring."""
    skip_if_sequential_too_slow(vectorized, n_starts=1, max_iter=3000)
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=5,
        max_iter=3000,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    true_solution = copy.deepcopy(solution)._compute_costs(instance.matrix)

    torch.testing.assert_close(solution.costs, true_solution.costs)
    torch.testing.assert_close(
        solution.costs, torch.sort(solution.costs).values, atol=0.0, rtol=0.0
    )


@pytest.mark.priority(5)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_energy_does_not_drift_on_large_batch(vectorized: bool) -> None:
    """On a large batch/instance, accumulated rounding error must not make costs drift.

    Reported costs must not drift from the true `x^T Q x`, for either path: both
    `_run_sequential` and `_run_vectorized` periodically recompute it exactly (see
    `_REFRESH_EVERY`), and recompute it once more from the final bitstrings before
    returning.
    """
    rng = torch_rng(64548)
    dataset = Dataset.from_random(1, 100, rng=rng)
    instance, _ = dataset[0]
    starts = bitstrings.rand(400, instance.size, rng=rng)

    solution = solving.simulated_annealing.solve(
        instance, starts=starts, max_iter=1000, vectorized=vectorized, rng=rng
    )
    check.is_true(solution.check_consistency(instance=instance, throw=False))


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_solution_is_internally_consistent(
    instance: Instance, vectorized: bool
) -> None:
    """The returned Solution must pass the full consistency check.

    Checks shapes, costs, sortedness, no duplicate bitstrings, positive integer counts,
    and probabilities matching normalised counts.
    """
    skip_if_sequential_too_slow(vectorized, n_starts=1, max_iter=3000)
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=5,
        max_iter=3000,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    check.is_true(solution.check_consistency(instance=instance, throw=True))


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_counts_sum_to_visits(instance: Instance, vectorized: bool) -> None:
    """Counts must be strictly positive integers.

    Their total must be at least the number of returned bitstrings.
    """
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=3,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    check.is_true(torch.all(solution.counts > 0).item())
    check.greater_equal(solution.counts.sum().item(), len(solution))


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_respects_top_k(instance: Instance, vectorized: bool) -> None:
    """The number of returned solutions never exceeds top_k."""
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=2,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    check.is_in(len(solution), [1, 2])


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_deterministic_with_seeded_rng(
    instance: Instance, vectorized: bool
) -> None:
    """Two runs with the same seed must produce identical solutions."""
    start = bitstrings.zeros(1, instance.size)

    solution_a = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=4,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(565111),
        vectorized=vectorized,
    )
    solution_b = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=4,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(565111),
        vectorized=vectorized,
    )

    torch.testing.assert_close(solution_a.bitstrings, solution_b.bitstrings)
    torch.testing.assert_close(solution_a.costs, solution_b.costs, atol=0.0, rtol=0.0)
    torch.testing.assert_close(solution_a.counts, solution_b.counts)
    torch.testing.assert_close(
        solution_a.probabilities, solution_b.probabilities, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_zero_max_iter_returns_start(
    instance: Instance, vectorized: bool
) -> None:
    """With max_iter=0, only the starting bitstring is returned."""
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=5,
        max_iter=0,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    check.equal(len(solution), 1)
    check.equal(solution[0].string, bitstring.to_string(start))
    check.equal(solution[0].count, 1)


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_zero_time_limit_returns_start(
    instance: Instance, vectorized: bool
) -> None:
    """An exhausted time budget stops the loop before any iteration runs."""
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=5,
        max_iter=1000,
        initial_temp=4.0,
        final_temp=0.05,
        time_limit=0.0,
        rng=rng,
        vectorized=vectorized,
    )

    check.equal(len(solution), 1)
    check.equal(solution[0].string, bitstring.to_string(start))


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_explicit_cooling_rate_used(
    instance: Instance, vectorized: bool
) -> None:
    """When cooling_rate is provided, final_temp is ignored.

    No error is raised even if final_temp is invalid (<= 0).
    """
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=3,
        max_iter=200,
        initial_temp=4.0,
        final_temp=-1.0,
        cooling_rate=0.9,
        rng=rng,
        vectorized=vectorized,
    )

    check.is_true(solution.check_consistency(instance=instance, throw=True))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": -1}, "top_k"),
        ({"initial_temp": 0.0}, "initial_temp"),
        ({"initial_temp": -1.0}, "initial_temp"),
        ({"final_temp": 0.0}, "final_temp"),
        ({"final_temp": -1.0}, "final_temp"),
        ({"cooling_rate": 0.0}, "cooling_rate"),
        ({"cooling_rate": 1.0}, "cooling_rate"),
        ({"cooling_rate": -0.5}, "cooling_rate"),
        ({"cooling_rate": 1.5}, "cooling_rate"),
    ],
)
def test_simulated_annealing_raises_on_invalid_arguments(kwargs: dict, match: str) -> None:
    start = bitstrings.zeros(1, instance_symmetric.size)

    with pytest.raises(ValueError, match=match):
        solving.simulated_annealing.solve(
            instance_symmetric, starts=start, rng=torch_rng(658), **kwargs
        )


def test_simulated_annealing_raises_on_starts_length_mismatch() -> None:
    """`starts` bitstrings shorter or longer than `instance.size` must raise, not crash."""
    short_start = bitstrings.zeros(1, instance_symmetric.size - 1)
    long_start = bitstrings.zeros(1, instance_symmetric.size + 1)

    with pytest.raises(ValueError, match=r"instance\.size"):
        solving.simulated_annealing.solve(instance_symmetric, starts=short_start)

    with pytest.raises(ValueError, match=r"instance\.size"):
        solving.simulated_annealing.solve(instance_symmetric, starts=long_start)


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_merge_false_returns_one_solution_per_start(
    instance: Instance, vectorized: bool
) -> None:
    """With merge=False, one Solution must be returned per row of `start`.

    They are returned in the same order, none of them merged with the others.
    """
    start = bitstrings.zeros(3, instance.size)
    rng = torch_rng(0)

    solutions = solving.simulated_annealing.solve(
        instance,
        starts=start,
        merge=False,
        top_k=2,
        max_iter=100,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        vectorized=vectorized,
    )

    check.equal(len(solutions), 3)
    for solution in solutions:
        check.is_true(solution.check_consistency(instance=instance, throw=True))


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_merge_true_matches_manual_concat_and_deduplicate(
    instance: Instance, vectorized: bool
) -> None:
    """merge=True (the default) must be equivalent to manually merging the per-start results.

    Equivalent to merging the merge=False per-start results via
    Solution.concat(...).deduplicate(), as documented on the function.
    """
    start = bitstrings.rand(4, instance.size, rng=torch_rng(574))
    top_k = 3

    merged_solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=top_k,
        max_iter=200,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(7874),
        vectorized=vectorized,
    )
    solutions = solving.simulated_annealing.solve(
        instance,
        starts=start,
        merge=False,
        top_k=top_k,
        max_iter=200,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(7874),
        vectorized=vectorized,
    )
    manually_merged = Solution.concat(solutions).deduplicate()

    check.is_true(merged_solution.check_consistency(instance=instance, throw=True))
    torch.testing.assert_close(merged_solution.bitstrings, manually_merged.bitstrings)
    torch.testing.assert_close(merged_solution.costs, manually_merged.costs, atol=0.0, rtol=0.0)
    torch.testing.assert_close(merged_solution.counts, manually_merged.counts)


@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_int_starts_generates_that_many_random_runs(vectorized: bool) -> None:
    """Passing an int for `starts` must generate that many uniformly random bitstrings.

    They are generated from `rng`, giving the same result as pre-generating them with
    bitstrings.rand from the same rng and passing them explicitly.
    """
    n_starts = 3

    rng_int = torch_rng(5821)
    solution = solving.simulated_annealing.solve(
        instance_symmetric,
        starts=n_starts,
        merge=False,
        top_k=2,
        max_iter=50,
        rng=rng_int,
        vectorized=vectorized,
    )

    rng_explicit = torch_rng(5821)
    expected_starts = bitstrings.rand(n_starts, instance_symmetric.size, rng=rng_explicit)
    expected = solving.simulated_annealing.solve(
        instance_symmetric,
        starts=expected_starts,
        merge=False,
        top_k=2,
        max_iter=50,
        rng=rng_explicit,
        vectorized=vectorized,
    )

    check.equal(len(solution), n_starts)
    for actual, exp in zip(solution, expected, strict=True):
        torch.testing.assert_close(actual.bitstrings, exp.bitstrings)
        torch.testing.assert_close(actual.costs, exp.costs, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_default_starts_is_one_random_start(vectorized: bool) -> None:
    """Omitting `starts` must default to a single uniformly random start.

    This produces exactly one run's worth of results.
    """
    solution = solving.simulated_annealing.solve(
        instance_symmetric,
        merge=False,
        top_k=1,
        max_iter=50,
        rng=torch_rng(0),
        vectorized=vectorized,
    )

    check.equal(len(solution), 1)
    check.is_true(solution[0].check_consistency(instance=instance_symmetric, throw=True))


def test_simulated_annealing_empty_start_merge_false_returns_empty_list() -> None:
    """An empty batch of starts must produce an empty list when merge=False.

    No runs are performed.
    """
    start = bitstrings.zeros(0, instance_symmetric.size)

    solutions = solving.simulated_annealing.solve(
        instance_symmetric, start, merge=False, rng=torch_rng(0)
    )

    check.equal(solutions, [])


def test_simulated_annealing_empty_start_merge_true_returns_empty_solution() -> None:
    """An empty batch of starts must produce an empty Solution when merge=True (default).

    No runs are performed.
    """
    start = bitstrings.zeros(0, instance_symmetric.size)

    solution = solving.simulated_annealing.solve(instance_symmetric, starts=start, rng=torch_rng(0))

    check.is_false(solution)


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_stats_per_run_sets_single_run_counts_to_one(
    instance: Instance, vectorized: bool
) -> None:
    """With a single run (one start), stats='per_run' must set every count to 1.

    Every returned bitstring's count is set to 1, regardless of how many iterations were
    actually spent at it. With multiple runs merged together, counts are not
    uniformly 1 -- see
    test_simulated_annealing_stats_per_run_merged_counts_reflect_run_agreement.
    """
    start = bitstrings.zeros(1, instance.size)
    rng = torch_rng(0)

    solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=3,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=rng,
        stats="per_run",
        vectorized=vectorized,
    )
    expected_counts = vectori.zeros(len(solution)).fill_(1)
    torch.testing.assert_close(solution.counts, expected_counts)


@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_stats_per_run_merged_counts_reflect_run_agreement(
    vectorized: bool,
) -> None:
    """With stats='per_run', top_k=1, and merge=True (default), each run contributes one count.

    Each run contributes a single bitstring with count 1; after merging, a bitstring's
    count is the number of runs that converged on it -- neither always 1 nor uniform
    across bitstrings.
    """
    skip_if_sequential_too_slow(vectorized, n_starts=8, max_iter=300)
    start = bitstrings.rand(8, instance_symmetric.size, rng=torch_rng(11))

    solution = solving.simulated_annealing.solve(
        instance_symmetric,
        starts=start,
        top_k=1,
        max_iter=300,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(0),
        stats="per_run",
        vectorized=vectorized,
    )

    check.equal(solution.counts.sum().item(), 8)
    check.is_true(torch.all(solution.counts >= 1).item())


@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_stats_per_run_top_k_one_merge_true_matches_manual_equivalent(
    vectorized: bool,
) -> None:
    """merge=True, top_k=1, stats='per_run' must be equivalent to a manual condensed run.

    Equivalent to running with merge=False, top_k>1, stats='full' (the default), then per
    start keeping only the best bitstring (truncate(1) -- each per-start Solution
    is already sorted by cost, so its first row is its best), concatenating
    with unit_counts=True (each surviving bitstring counts as a single vote,
    matching stats='per_run'), and deduplicating.

    This also shows how to get the condensed, single-best-per-run result that
    most other optimization libraries return by default, while still running
    with stats='full' to keep the complete per-run results available if
    needed.
    """
    skip_if_sequential_too_slow(vectorized, n_starts=8, max_iter=300)
    start = bitstrings.rand(8, instance_symmetric.size, rng=torch_rng(1350))

    per_run_solution = solving.simulated_annealing.solve(
        instance_symmetric,
        starts=start,
        max_iter=300,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(0),
        vectorized=vectorized,
    )

    solutions = solving.simulated_annealing.solve(
        instance_symmetric,
        starts=start,
        merge=False,
        top_k=3,
        max_iter=300,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(0),
        stats="full",
        vectorized=vectorized,
    )
    manually_equivalent = Solution.concat(
        [solution.truncate(1) for solution in solutions], unit_counts=True
    ).deduplicate()

    torch.testing.assert_close(per_run_solution.bitstrings, manually_equivalent.bitstrings)
    torch.testing.assert_close(
        per_run_solution.costs, manually_equivalent.costs, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(per_run_solution.counts, manually_equivalent.counts)
    torch.testing.assert_close(
        per_run_solution.probabilities, manually_equivalent.probabilities, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("instance", instances, ids=instance_ids)
@pytest.mark.parametrize("vectorized", vectorized_params)
def test_simulated_annealing_stats_per_run_is_default(instance: Instance, vectorized: bool) -> None:
    """Omitting stats must be equivalent to passing stats='per_run' explicitly."""
    start = bitstrings.zeros(1, instance.size)

    default_solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=3,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(23),
        vectorized=vectorized,
    )
    explicit_per_run_solution = solving.simulated_annealing.solve(
        instance,
        starts=start,
        top_k=3,
        max_iter=500,
        initial_temp=4.0,
        final_temp=0.05,
        rng=torch_rng(23),
        stats="per_run",
        vectorized=vectorized,
    )

    torch.testing.assert_close(default_solution.counts, explicit_per_run_solution.counts)


def test_simulated_annealing_stats_per_run_top_k_above_one_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """stats='per_run' with top_k > 1 must emit an info-level log about merged counts.

    The log notes that merged counts won't simply be 1 per run.
    """
    start = bitstrings.zeros(1, instance_symmetric.size)

    with caplog.at_level("INFO", logger="qubosolver.solving.classical.simulated_annealing"):
        solving.simulated_annealing.solve(
            instance_symmetric,
            starts=start,
            top_k=2,
            max_iter=50,
            initial_temp=4.0,
            final_temp=0.05,
            rng=torch_rng(0),
            stats="per_run",
        )

    check.is_true(any("per_run" in record.message for record in caplog.records))


def test_simulated_annealing_stats_per_run_top_k_one_does_not_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """stats='per_run' with top_k=1 (the intended, unambiguous usage) must not log.

    It must not emit the top_k > 1 caveat log.
    """
    start = bitstrings.zeros(1, instance_symmetric.size)

    with caplog.at_level("INFO", logger="qubosolver.solving.classical.simulated_annealing"):
        solving.simulated_annealing.solve(
            instance_symmetric,
            starts=start,
            top_k=1,
            max_iter=50,
            initial_temp=4.0,
            final_temp=0.05,
            rng=torch_rng(12),
            stats="per_run",
        )

    check.equal(caplog.records, [])


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason=(
        "typing_extensions.get_overloads only sees overloads registered via "
        "typing.overload starting on Python 3.11 (it delegates to typing's "
        "own registry); on 3.10 it maintains a separate registry that only "
        "typing_extensions.overload writes into."
    ),
)
def test_simulated_annealing_overloads_match_implementation_signature() -> None:
    """Every @overload stub of simulated_annealing must match the real implementation exactly.

    Each stub must declare exactly the same parameters, with the same defaults, as the
    real implementation, so adding/removing/renaming a parameter -- or changing its
    default -- on only one of the three cannot silently drift from the others.

    `merge` is exempted from the default-value check: it intentionally has
    no default in the merge=False overload (it must be passed explicitly to
    select that overload), unlike the impl and the merge=True overload
    where it defaults to True.
    """
    from qubosolver.solving.classical.simulated_annealing import solve

    impl_params = inspect.signature(solve).parameters
    overloads = get_overloads(solve)

    check.greater(len(overloads), 0)
    for overload_func in overloads:
        overload_params = inspect.signature(overload_func).parameters
        check.equal(overload_params.keys(), impl_params.keys())

        for name, impl_param in impl_params.items():
            if name == "merge":
                continue
            overload_param = overload_params[name]
            # rng's default is a single Generator built once at import time
            # and shared by every caller that omits it; comparing by identity
            # verifies all three signatures point at that same instance
            # rather than each constructing their own (which would silently
            # break the "one shared default rng" invariant).
            if name == "rng":
                check.is_(overload_param.default, impl_param.default)
            else:
                check.equal(overload_param.default, impl_param.default)


def test_simulated_annealing_overload_return_types_are_statically_correct() -> None:
    """Static-typing check (evaluated by mypy, not at runtime) on merge's return type.

    merge=True (default) must be inferred as Solution, and merge=False as list[Solution].
    This function's body never actually executes assertions at runtime --
    assert_type is a no-op there -- its only purpose is to be type-checked.
    """
    start = bitstrings.zeros(1, instance_symmetric.size)

    default_result = solving.simulated_annealing.solve(instance_symmetric, starts=start)
    assert_type(default_result, Solution)

    explicit_merge_result = solving.simulated_annealing.solve(
        instance_symmetric, starts=start, merge=True
    )
    assert_type(explicit_merge_result, Solution)

    unmerged_result = solving.simulated_annealing.solve(
        instance_symmetric, starts=start, merge=False
    )
    assert_type(unmerged_result, list[Solution])
