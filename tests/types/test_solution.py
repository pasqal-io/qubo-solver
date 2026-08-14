from __future__ import annotations

import copy

import pytest
import pytest_check as check
import torch

from qubosolver import Instance, Solution, matrix, bitstrings, vector, vectori


@pytest.fixture
def instance() -> Instance:
    Q = matrix.tensor([[1.0, -1.0], [-1.0, 2.0]])
    return Instance(matrix=Q)


def _assert_valid(solution: Solution, instance: Instance) -> None:
    check.is_true(solution.check_consistency(instance))
    check.is_true(solution.check_consistency(instance, throw=True))


def _assert_invalid(solution: Solution, instance: Instance) -> None:
    check.is_false(solution.check_consistency(instance))
    with pytest.raises(AssertionError):
        solution.check_consistency(instance, throw=True)


def test_valid_solution_passes(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    _assert_valid(solution, instance)


def test_empty_solution_against_empty_qubo_is_valid() -> None:
    _assert_valid(Solution(), Instance())


def test_empty_solution_against_non_empty_qubo_is_invalid(instance: Instance) -> None:
    _assert_invalid(Solution(), instance)


def test_wrong_bitstring_width_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0, 1], [0, 1, 0]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    _assert_invalid(solution, instance)


@pytest.mark.parametrize("missing", ["costs", "counts", "probabilities"])
def test_missing_field_is_invalid(instance: Instance, missing: str) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    setattr(solution, missing, torch.tensor([], dtype=getattr(solution, missing).dtype))
    _assert_invalid(solution, instance)


def test_wrong_costs_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 999.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    _assert_invalid(solution, instance)


def test_unsorted_costs_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[0, 1], [1, 0]]),
        costs=vector.tensor([2.0, 1.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    _assert_invalid(solution, instance)


def test_duplicate_bitstrings_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [1, 0]]),
        costs=vector.tensor([1.0, 1.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    _assert_invalid(solution, instance)


def test_non_positive_counts_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([0, 1]),
        probabilities=vector.tensor([0.0, 1.0]),
    )
    _assert_invalid(solution, instance)


def test_probabilities_inconsistent_with_counts_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    _assert_invalid(solution, instance)


def test_non_binary_bitstrings_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    solution.bitstrings[0, 0] = 2
    _assert_invalid(solution, instance)


def test_non_integer_counts_is_invalid(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([1, 1]),
        probabilities=vector.tensor([0.5, 0.5]),
    )
    solution.counts = torch.tensor([1.5, 1.5])
    _assert_invalid(solution, instance)


def test_deduplicate_sums_counts_for_duplicate_bitstrings(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1], [1, 0]]),
        costs=vector.tensor([1.0, 2.0, 1.0]),
        counts=vectori.tensor([2, 1, 3]),
        probabilities=vector.tensor([1 / 3, 1 / 6, 1 / 2]),
    )
    solution.deduplicate()
    _assert_valid(solution, instance)
    assert len(solution) == 2

    s0 = solution[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 5)
    check.almost_equal(s0.probability, 5 / 6)

    s1 = solution[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.almost_equal(s1.probability, 1 / 6)


def test_deduplicate_keeps_minimum_cost_for_duplicate_bitstrings() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [1, 0], [1, 0]]),
        costs=vector.tensor([3.0, 1.0, 2.0]),
        counts=vectori.tensor([1, 1, 1]),
        probabilities=vector.tensor([1 / 3, 1 / 3, 1 / 3]),
    )
    solution.deduplicate()
    assert len(solution) == 1

    s0 = solution[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)


def test_deduplicate_no_duplicates_is_noop(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    solution.deduplicate()
    _assert_valid(solution, instance)
    assert len(solution) == 2

    s0 = solution[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)
    check.equal(s0.probability, 0.75)

    s1 = solution[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.equal(s1.probability, 0.25)


def test_deduplicate_empty_solution_is_noop() -> None:
    solution = Solution()
    solution.deduplicate()
    _assert_valid(solution, Instance())
    check.is_false(solution)


def test_deduplicate_missing_counts_is_skipped() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1], [1, 0]]),
        costs=vector.tensor([1.0, 2.0, 1.0]),
    )
    solution.deduplicate()
    check.equal(len(solution), 2)
    check.equal(solution.counts.numel(), 0)
    check.equal(solution.probabilities.numel(), 0)


def test_deduplicate_missing_costs_is_skipped() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1], [1, 0]]),
        counts=vectori.tensor([2, 1, 3]),
    )
    solution.deduplicate()
    assert len(solution) == 2
    check.equal(solution.costs.numel(), 0)

    s0 = solution[0]
    check.equal(s0.string, "01")
    check.equal(s0.count, 1)
    check.almost_equal(s0.probability, 1 / 6)

    s1 = solution[1]
    check.equal(s1.string, "10")
    check.equal(s1.count, 5)
    check.almost_equal(s1.probability, 5 / 6)


def test_concat_disjoint_bitstrings_concatenates(instance: Instance) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([3]),
        probabilities=vector.tensor([1.0]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([2.0]),
        counts=vectori.tensor([1]),
        probabilities=vector.tensor([1.0]),
    )
    concatenated = Solution.concat([a, b]).deduplicate()
    _assert_valid(concatenated, instance)
    assert len(concatenated) == 2

    s0 = concatenated[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)
    check.equal(s0.probability, 0.75)

    s1 = concatenated[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.equal(s1.probability, 0.25)


def test_concat_three_solutions(instance: Instance) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([3]),
        probabilities=vector.tensor([1.0]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([2.0]),
        counts=vectori.tensor([1]),
        probabilities=vector.tensor([1.0]),
    )
    c = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([2]),
        probabilities=vector.tensor([1.0]),
    )
    concatenated = Solution.concat([a, b, c]).deduplicate()
    _assert_valid(concatenated, instance)
    assert len(concatenated) == 2

    s0 = concatenated[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 5)
    check.almost_equal(s0.probability, 5 / 6)

    s1 = concatenated[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.almost_equal(s1.probability, 1 / 6)


def test_concat_keeps_duplicate_bitstrings_as_separate_rows(instance: Instance) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([2]),
        probabilities=vector.tensor([1.0]),
    )
    concatenated = Solution.concat([a, b])
    check.equal(len(concatenated), 3)
    _assert_invalid(concatenated, instance)


def test_concat_then_deduplicate_sums_counts_for_overlapping_bitstrings(
    instance: Instance,
) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([2]),
        probabilities=vector.tensor([1.0]),
    )
    merged = Solution.concat([a, b]).deduplicate()
    _assert_valid(merged, instance)
    assert len(merged) == 2

    s0 = merged[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 5)
    check.almost_equal(s0.probability, 5 / 6)

    s1 = merged[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.almost_equal(s1.probability, 1 / 6)


def test_concat_unit_counts_sets_counts_to_one_before_dedup(instance: Instance) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 7]),
        probabilities=vector.tensor([3 / 10, 7 / 10]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([99]),
        probabilities=vector.tensor([1.0]),
    )
    merged = Solution.concat([a, b], unit_counts=True).deduplicate()
    _assert_valid(merged, instance)
    assert len(merged) == 2

    s0 = merged[0]
    check.equal(s0.string, "10")
    check.equal(s0.count, 2)
    check.almost_equal(s0.probability, 2 / 3)

    s1 = merged[1]
    check.equal(s1.string, "01")
    check.equal(s1.count, 1)
    check.almost_equal(s1.probability, 1 / 3)


def test_concat_unit_counts_without_counts_leaves_probabilities_empty() -> None:
    a = Solution(bitstrings=bitstrings.tensor([[1, 0]]), costs=vector.tensor([1.0]))
    b = Solution(bitstrings=bitstrings.tensor([[0, 1]]), costs=vector.tensor([2.0]))
    merged = Solution.concat([a, b], unit_counts=True)
    check.equal(merged.counts.tolist(), [1, 1])
    check.equal(merged.probabilities.numel(), 0)


def test_concat_accepts_generator(instance: Instance) -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
        counts=vectori.tensor([3]),
        probabilities=vector.tensor([1.0]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([2.0]),
        counts=vectori.tensor([1]),
        probabilities=vector.tensor([1.0]),
    )
    concatenated = Solution.concat(s for s in (a, b)).deduplicate()
    _assert_valid(concatenated, instance)
    assert len(concatenated) == 2

    s0 = concatenated[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)
    check.equal(s0.probability, 0.75)

    s1 = concatenated[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.equal(s1.probability, 0.25)


def test_concat_both_empty_is_empty() -> None:
    concatenated = Solution.concat([Solution(), Solution()])
    _assert_valid(concatenated, Instance())
    check.equal(len(concatenated), 0)


def test_concat_empty_sequence_is_empty() -> None:
    concatenated = Solution.concat([])
    _assert_valid(concatenated, Instance())
    check.equal(len(concatenated), 0)


@pytest.mark.parametrize("empty_first", [True, False])
def test_concat_one_empty_returns_other(instance: Instance, empty_first: bool) -> None:
    nonempty = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    a, b = (Solution(), nonempty) if empty_first else (nonempty, Solution())
    concatenated = Solution.concat([a, b])
    _assert_valid(concatenated, instance)
    assert len(concatenated) == 2

    s0 = concatenated[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)
    check.equal(s0.probability, 0.75)

    s1 = concatenated[1]
    check.equal(s1.string, "01")
    check.equal(s1.cost, 2.0)
    check.equal(s1.count, 1)
    check.equal(s1.probability, 0.25)


def test_concat_mixed_populated_and_empty_costs_raises() -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        counts=vectori.tensor([2]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([1.5]),
        counts=vectori.tensor([3]),
    )
    with pytest.raises(ValueError):
        Solution.concat([a, b])


def test_truncate_keeps_first_k_rows(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    solution.truncate(1)
    _assert_valid(solution, instance)
    assert len(solution) == 1

    s0 = solution[0]
    check.equal(s0.string, "10")
    check.equal(s0.cost, 1.0)
    check.equal(s0.count, 3)
    check.equal(s0.probability, 1.0)


def test_truncate_recomputes_probabilities_from_counts() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1], [0, 0]]),
        counts=vectori.tensor([3, 1, 4]),
        probabilities=vector.tensor([3 / 8, 1 / 8, 4 / 8]),
    )
    solution.truncate(2)
    assert len(solution) == 2
    check.almost_equal(solution[0].probability, 0.75)
    check.almost_equal(solution[1].probability, 0.25)


def test_truncate_k_greater_than_len_is_noop(instance: Instance) -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    solution.truncate(10)
    _assert_valid(solution, instance)
    assert len(solution) == 2


def test_truncate_missing_costs_and_counts_is_skipped() -> None:
    solution = Solution(bitstrings=bitstrings.tensor([[1, 0], [0, 1], [0, 0]]))
    solution.truncate(2)
    assert len(solution) == 2
    check.equal(solution.costs.numel(), 0)
    check.equal(solution.counts.numel(), 0)
    check.equal(solution.probabilities.numel(), 0)


def test_truncate_empty_solution_is_noop() -> None:
    solution = Solution()
    solution.truncate(1)
    _assert_valid(solution, Instance())
    check.is_false(solution)


def test_truncate_probabilities_without_counts_raises() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1], [0, 0]]),
        probabilities=vector.tensor([0.375, 0.125, 0.5]),
    )
    with pytest.raises(ValueError):
        solution.truncate(2)


def test_deepcopy_is_independent_of_original() -> None:
    solution = Solution(
        bitstrings=bitstrings.tensor([[1, 0], [0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([3, 1]),
        probabilities=vector.tensor([0.75, 0.25]),
    )
    cloned = copy.deepcopy(solution)

    cloned.bitstrings[0, 0] = 0
    cloned.costs[0] = 999.0
    cloned.counts[0] = 42
    cloned.probabilities[0] = 0.1

    check.equal(solution[0].string, "10")
    check.equal(solution[0].cost, 1.0)
    check.equal(solution[0].count, 3)
    check.equal(solution[0].probability, 0.75)


def test_concat_mixed_populated_and_empty_counts_raises() -> None:
    a = Solution(
        bitstrings=bitstrings.tensor([[1, 0]]),
        costs=vector.tensor([1.0]),
    )
    b = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([1.5]),
        counts=vectori.tensor([3]),
    )
    with pytest.raises(ValueError):
        Solution.concat([a, b])
