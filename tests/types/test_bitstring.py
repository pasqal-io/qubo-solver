# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import random

import numpy as np
import pytest
import pytest_check as check

from qubosolver import bitstring, numpy_rng


def test_from_string_creates_bitstring_from_empty_string() -> None:
    result = bitstring.from_string("")
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_from_string_creates_bitstring_from_single_zero() -> None:
    result = bitstring.from_string("0")
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(result, np.asarray([0], dtype=np.int8), rtol=0, strict=True)


def test_from_string_creates_bitstring_from_single_one() -> None:
    result = bitstring.from_string("1")
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(result, np.asarray([1], dtype=np.int8), rtol=0, strict=True)


def test_from_string_creates_bitstring_from_multi_char_string() -> None:
    result = bitstring.from_string("01101")
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(
        result, np.asarray([0, 1, 1, 0, 1], dtype=np.int8), rtol=0, strict=True
    )


def test_from_string_roundtrips_with_to_string() -> None:
    original = "1011001011100101"
    result = bitstring.to_string(bitstring.from_string(original))
    check.equal(result, original)


@pytest.mark.parametrize("seed", [42, 137, 9991])
@pytest.mark.parametrize("length", [0, 1, 2, 5, 10, 50, 100])
def test_from_string_roundtrips_with_to_string_random(seed: int, length: int) -> None:
    rng = random.Random(seed)  # noqa: S311 (seeded test data, not cryptographic)
    original = "".join(rng.choice("01") for _ in range(length))
    roundtripped = bitstring.to_string(bitstring.from_string(original))
    check.equal(roundtripped, original)


def test_to_string_returns_empty_string_for_empty_tensor() -> None:
    empty = np.asarray([], dtype=np.int8)
    result = bitstring.to_string(empty)
    check.equal(result, "")


def test_to_string_returns_zero_for_single_zero_element() -> None:
    single = np.asarray([0], dtype=np.int8)
    result = bitstring.to_string(single)
    check.equal(result, "0")


def test_to_string_returns_one_for_single_one_element() -> None:
    single = np.asarray([1], dtype=np.int8)
    result = bitstring.to_string(single)
    check.equal(result, "1")


def test_to_string_returns_correct_string_for_multi_element_tensor() -> None:
    bs = np.asarray([0, 1, 1, 0, 1], dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "01101")


def test_to_string_returns_all_zeros_for_zeros_bitstring() -> None:
    bs = np.zeros(5, dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "00000")


def test_to_string_returns_all_ones_for_ones_bitstring() -> None:
    bs = np.asarray([1, 1, 1, 1, 1], dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "11111")


def test_to_string_returns_correct_string_for_long_bitstring() -> None:
    bs = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "1011001011100101")


def test_to_string_converts_bitstring_tensor_to_string() -> None:
    bs = np.asarray([1, 0, 0, 1, 1], dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "10011")


def test_to_string_converts_as_tensor_bitstring_to_string() -> None:
    source = np.array([1, 0, 1, 1, 0], dtype=np.int32)
    bs = np.asarray(source, dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(result, "10110")


def test_to_string_length_matches_number_of_elements() -> None:
    data = [1, 0, 1, 0, 1, 1, 0]
    bs = np.asarray(data, dtype=np.int8)
    result = bitstring.to_string(bs)
    check.equal(len(result), len(data))


def test_rand_creates_int8_vector_of_requested_length() -> None:
    n = 20
    result = bitstring.rand(n, rng=numpy_rng(0))
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (n,))


def test_rand_creates_empty_vector_when_n_is_zero() -> None:
    result = bitstring.rand(0, rng=numpy_rng(0))
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0,))


def test_rand_only_contains_zeros_and_ones() -> None:
    result = bitstring.rand(200, rng=numpy_rng(0))
    check.is_true(np.all((result == 0) | (result == 1)).item())


def test_rand_is_deterministic_with_seeded_rng() -> None:
    a = bitstring.rand(50, rng=numpy_rng(1234))
    b = bitstring.rand(50, rng=numpy_rng(1234))
    np.testing.assert_allclose(a, b, rtol=0, strict=True)


def test_rand_differs_across_seeds() -> None:
    a = bitstring.rand(50, rng=numpy_rng(1))
    b = bitstring.rand(50, rng=numpy_rng(2))
    check.is_false(np.array_equal(a, b))


def test_round_converts_exact_values() -> None:
    result = bitstring.round([1.0, 0.0, 1.0, 0.0])
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(result, np.asarray([1, 0, 1, 0], dtype=np.int8), rtol=0, strict=True)


def test_round_rounds_near_integral_values() -> None:
    result = bitstring.round([0.9999999, 1e-9, -0.0, 1.0000001])
    np.testing.assert_allclose(result, np.asarray([1, 0, 0, 1], dtype=np.int8), rtol=0, strict=True)


def test_round_accepts_values_just_inside_the_tolerance() -> None:
    result = bitstring.round([9e-7, 1.0 - 9e-7])
    np.testing.assert_allclose(result, np.asarray([0, 1], dtype=np.int8), rtol=0, strict=True)


def test_round_accepts_a_custom_tolerance() -> None:
    result = bitstring.round([0.999, 0.001], atol=1e-2)
    np.testing.assert_allclose(result, np.asarray([1, 0], dtype=np.int8), rtol=0, strict=True)


def test_round_returns_empty_tensor_for_empty_input() -> None:
    result = bitstring.round([])
    check.equal(result.dtype, np.int8)
    check.equal(result.shape, (0,))


def test_round_accepts_a_float_tensor() -> None:
    source = np.array([1.0, 0.0, 0.9999999])
    result = bitstring.round(source)
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(result, np.asarray([1, 0, 1], dtype=np.int8), rtol=0, strict=True)


def test_round_accepts_a_numpy_float_array() -> None:
    source = np.array([0.0, 1.0, 1e-9])
    result = bitstring.round(source)
    check.equal(result.dtype, np.int8)
    np.testing.assert_allclose(result, np.asarray([0, 1, 0], dtype=np.int8), rtol=0, strict=True)


def test_round_raises_on_fractional_value() -> None:
    with pytest.raises(ValueError, match="within 1e-06 of 0 or 1"):
        bitstring.round([1.0, 0.5, 0.0])


def test_round_raises_on_value_just_outside_the_tolerance() -> None:
    with pytest.raises(ValueError, match="of 0 or 1"):
        bitstring.round([1.0 - 2e-6])


def test_round_raises_on_out_of_range_value() -> None:
    with pytest.raises(ValueError, match="of 0 or 1"):
        bitstring.round([0.0, 2.0])


def test_round_raises_on_negative_value() -> None:
    with pytest.raises(ValueError, match="of 0 or 1"):
        bitstring.round([-1.0, 0.0])


def test_round_raises_when_custom_tolerance_is_tighter_than_the_error() -> None:
    with pytest.raises(ValueError, match="within 1e-09 of 0 or 1"):
        bitstring.round([0.9999999], atol=1e-9)


def test_round_error_message_reports_the_offending_values() -> None:
    with pytest.raises(ValueError, match=r"got \[0\.5\]"):
        bitstring.round([1.0, 0.5, 0.0])


@pytest.mark.parametrize("seed", [73, 256, 8401])
@pytest.mark.parametrize("length", [1, 2, 5, 10, 50])
def test_round_matches_from_string_on_perturbed_bits(seed: int, length: int) -> None:
    rng = random.Random(seed)  # noqa: S311 (seeded test data, not cryptographic)
    string = "".join(rng.choice("01") for _ in range(length))
    perturbed = [int(c) + rng.uniform(-1e-7, 1e-7) for c in string]
    np.testing.assert_allclose(
        bitstring.round(perturbed), bitstring.from_string(string), rtol=0, strict=True
    )
