from __future__ import annotations

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
