# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import random

import numpy as np
import pytest
import pytest_check as check

from qubosolver import bitstrings, numpy_rng


def test_from_strings_creates_tensor_from_single_string() -> None:
    result = bitstrings.from_strings(["101"])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (1, 3))
    np.testing.assert_allclose(result, np.asarray([[1, 0, 1]], dtype=np.int8), rtol=0, strict=True)


def test_from_strings_creates_tensor_from_multiple_strings() -> None:
    result = bitstrings.from_strings(["011", "101", "000"])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (3, 3))
    np.testing.assert_allclose(
        result, np.asarray([[0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=np.int8), rtol=0, strict=True
    )


def test_from_strings_creates_all_zeros_tensor() -> None:
    result = bitstrings.from_strings(["0000", "0000", "0000"])
    check.equal(result.shape, (3, 4))
    np.testing.assert_allclose(result, np.zeros((3, 4), dtype=np.int8), rtol=0, strict=True)


def test_from_strings_creates_all_ones_tensor() -> None:
    result = bitstrings.from_strings(["111", "111"])
    np.testing.assert_allclose(
        result, np.asarray([[1, 1, 1], [1, 1, 1]], dtype=np.int8), rtol=0, strict=True
    )


def test_from_strings_roundtrips_with_to_strings() -> None:
    original = ["01101", "10010", "11111", "00000"]
    roundtripped = bitstrings.to_strings(bitstrings.from_strings(original))
    check.equal(roundtripped, original)


def test_from_strings_raises_on_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        bitstrings.from_strings(["01", "101", "0"])


def test_from_strings_returns_empty_tensor_for_empty_list() -> None:
    result = bitstrings.from_strings([])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0, 0))


@pytest.mark.parametrize("seed", [73, 256, 8401])
@pytest.mark.parametrize("count", [1, 2, 5, 10])
@pytest.mark.parametrize("length", [1, 2, 5, 10, 50])
def test_from_strings_roundtrips_with_to_strings_random(seed: int, count: int, length: int) -> None:
    rng = random.Random(seed)  # noqa: S311 (seeded test data, not cryptographic)
    original = ["".join(rng.choice("01") for _ in range(length)) for _ in range(count)]
    roundtripped = bitstrings.to_strings(bitstrings.from_strings(original))
    check.equal(roundtripped, original)


def test_to_strings_returns_empty_list_for_empty_tensor() -> None:
    empty = np.zeros((0, 5), dtype=np.int8)
    result = bitstrings.to_strings(empty)
    check.equal(result, [])


def test_to_strings_returns_single_string_for_single_row() -> None:
    bs = np.asarray([[1, 0, 1]], dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["101"])


def test_to_strings_returns_correct_strings_for_multiple_rows() -> None:
    bs = np.asarray([[0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["011", "101", "000"])


def test_to_strings_returns_all_zeros_strings() -> None:
    bs = np.zeros((3, 4), dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["0000", "0000", "0000"])


def test_to_strings_returns_all_ones_strings() -> None:
    bs = np.asarray([[1, 1, 1], [1, 1, 1]], dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["111", "111"])


def test_to_strings_length_matches_number_of_rows() -> None:
    bs = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(len(result), 4)


def test_to_strings_as_tensor_converts_correctly() -> None:
    source = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    bs = np.asarray(source, dtype=np.int8)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["101", "010"])


def test_rand_creates_int8_matrix_of_requested_shape() -> None:
    count, n_bits = 6, 20
    result = bitstrings.rand(count, n_bits, rng=numpy_rng(0))
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (count, n_bits))


def test_rand_creates_empty_matrix_when_count_is_zero() -> None:
    result = bitstrings.rand(0, 5, rng=numpy_rng(0))
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0, 5))


def test_rand_creates_empty_matrix_when_n_bits_is_zero() -> None:
    result = bitstrings.rand(5, 0, rng=numpy_rng(0))
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (5, 0))


def test_rand_only_contains_zeros_and_ones() -> None:
    result = bitstrings.rand(20, 20, rng=numpy_rng(0))
    check.is_true(np.all((result == 0) | (result == 1)).item())


def test_rand_is_deterministic_with_seeded_rng() -> None:
    a = bitstrings.rand(10, 50, rng=numpy_rng(1234))
    b = bitstrings.rand(10, 50, rng=numpy_rng(1234))
    np.testing.assert_allclose(a, b, rtol=0, strict=True)


def test_rand_differs_across_seeds() -> None:
    a = bitstrings.rand(10, 50, rng=numpy_rng(1))
    b = bitstrings.rand(10, 50, rng=numpy_rng(2))
    check.is_false(np.array_equal(a, b))


def test_rand_rows_are_independent_not_identical() -> None:
    result = bitstrings.rand(10, 30, rng=numpy_rng(0))
    check.is_false(np.all(result == result[0]).item())


def test_round_converts_exact_values() -> None:
    result = bitstrings.round([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (2, 3))
    np.testing.assert_allclose(
        result, np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.int8), rtol=0, strict=True
    )


def test_round_rounds_near_integral_values() -> None:
    result = bitstrings.round([[0.9999999, 1e-9], [-0.0, 1.0000001]])
    np.testing.assert_allclose(
        result, np.asarray([[1, 0], [0, 1]], dtype=np.int8), rtol=0, strict=True
    )


def test_round_accepts_a_custom_tolerance() -> None:
    result = bitstrings.round([[0.999, 0.001]], atol=1e-2)
    np.testing.assert_allclose(result, np.asarray([[1, 0]], dtype=np.int8), rtol=0, strict=True)


def test_round_returns_empty_tensor_for_empty_list() -> None:
    result = bitstrings.round([])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0, 0))


def test_round_accepts_a_2d_float_tensor() -> None:
    source = np.array([[1.0, 0.0], [0.9999999, 1e-9]])
    result = bitstrings.round(source)
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(
        result, np.asarray([[1, 0], [1, 0]], dtype=np.int8), rtol=0, strict=True
    )


def test_round_accepts_a_2d_numpy_float_array() -> None:
    source = np.array([[0.0, 1.0], [1.0, 1e-9]])
    result = bitstrings.round(source)
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(
        result, np.asarray([[0, 1], [1, 0]], dtype=np.int8), rtol=0, strict=True
    )


def test_round_raises_on_ragged_rows() -> None:
    with pytest.raises(ValueError, match="inhomogeneous shape"):
        bitstrings.round([[1.0, 0.0], [1.0, 0.0, 1.0]])


def test_round_raises_on_fractional_value() -> None:
    with pytest.raises(ValueError, match="of 0 or 1"):
        bitstrings.round([[1.0, 0.0], [0.5, 1.0]])


def test_round_raises_when_custom_tolerance_is_tighter_than_the_error() -> None:
    with pytest.raises(ValueError, match="within 1e-09 of 0 or 1"):
        bitstrings.round([[0.9999999, 0.0]], atol=1e-9)


@pytest.mark.parametrize("seed", [73, 256, 8401])
@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("length", [1, 2, 5, 10, 50])
def test_round_matches_from_strings_on_perturbed_bits(seed: int, count: int, length: int) -> None:
    rng = random.Random(seed)  # noqa: S311 (seeded test data, not cryptographic)
    strings = ["".join(rng.choice("01") for _ in range(length)) for _ in range(count)]
    perturbed = [[int(c) + rng.uniform(-1e-7, 1e-7) for c in s] for s in strings]
    np.testing.assert_allclose(
        bitstrings.round(perturbed), bitstrings.from_strings(strings), rtol=0, strict=True
    )
