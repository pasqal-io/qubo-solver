from __future__ import annotations

import pytest
import pytest_check as check

from qubosolver.types.label import Labelling, _to_callable


def test_callable_input_returns_same_function() -> None:
    """Test that callable input is returned unchanged."""

    def label_func(i: int) -> str:
        return f"var_{i}"

    result = _to_callable(label_func)
    check.equal(result, label_func)
    check.equal(result(0), "var_0")
    check.equal(result(5), "var_5")


def test_sequence_input_creates_indexing_function() -> None:
    """Test that sequence input creates a function that indexes into it."""
    labels = ["x", "y", "z"]
    result = _to_callable(labels)

    check.equal(result(0), "x")
    check.equal(result(1), "y")
    check.equal(result(2), "z")


def test_list_input_creates_indexing_function() -> None:
    """Test that list input works correctly."""
    labels = ["alpha", "beta", "gamma"]
    result = _to_callable(labels)

    check.equal(result(0), "alpha")
    check.equal(result(1), "beta")
    check.equal(result(2), "gamma")


def test_tuple_input_creates_indexing_function() -> None:
    """Test that tuple input works correctly."""
    labels = ("a", "b", "c", "d")
    result = _to_callable(labels)

    check.equal(result(0), "a")
    check.equal(result(1), "b")
    check.equal(result(2), "c")
    check.equal(result(3), "d")


def test_sequence_index_error_raised_for_out_of_bounds() -> None:
    """Test that IndexError is raised for out-of-bounds sequence access."""
    labels = ["x", "y"]
    result = _to_callable(labels)

    # Valid indices should work
    check.equal(result(0), "x")
    check.equal(result(1), "y")

    # Out of bounds should raise IndexError
    with pytest.raises(IndexError):
        result(2)

    with pytest.raises(IndexError):
        result(-3)


def test_empty_sequence_raises_index_error() -> None:
    """Test that empty sequence raises IndexError for any index."""
    labels: list[str] = []
    result = _to_callable(labels)

    with pytest.raises(IndexError):
        result(0)

    with pytest.raises(IndexError):
        result(-1)


def test_lambda_function_works_correctly() -> None:
    """Test that lambda functions work as expected."""
    label_func = lambda i: f"node_{i}"  # noqa: E731
    result = _to_callable(label_func)

    check.equal(result(0), "node_0")
    check.equal(result(10), "node_10")
    check.equal(result(999), "node_999")


def test_complex_labelling_function() -> None:
    """Test a more complex labelling function."""

    def complex_labeller(i: int) -> str:
        if i == 0:
            return "root"
        elif i < 10:
            return f"single_{i}"
        else:
            return f"multi_{i:02d}"

    result = _to_callable(complex_labeller)

    check.equal(result(0), "root")
    check.equal(result(5), "single_5")
    check.equal(result(15), "multi_15")
    check.equal(result(100), "multi_100")


def test_sequence_types_are_valid_labelling() -> None:
    """Test that various sequence types satisfy the Labelling type."""
    # These should all be valid Labelling types
    list_labels: Labelling = ["a", "b", "c"]
    tuple_labels: Labelling = ("x", "y", "z")

    # Test they work with _to_callable
    list_func = _to_callable(list_labels)
    tuple_func = _to_callable(tuple_labels)

    check.equal(list_func(0), "a")
    check.equal(tuple_func(1), "y")


def test_callable_types_are_valid_labelling() -> None:
    """Test that callable types satisfy the Labelling type."""
    # These should all be valid Labelling types
    lambda_labels: Labelling = lambda i: f"var_{i}"  # noqa: E731

    def function_labels(i: int) -> str:
        return f"func_{i}"

    func_labels: Labelling = function_labels

    # Test they work with _to_callable
    lambda_func = _to_callable(lambda_labels)
    func_func = _to_callable(func_labels)

    check.equal(lambda_func(0), "var_0")
    check.equal(func_func(5), "func_5")
