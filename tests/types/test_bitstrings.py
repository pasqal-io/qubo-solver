from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import pytest_check as check

from qubosolver import bitstring, bitstrings, torch_rng


def test_dtype_returns_int8() -> None:
    check.equal(bitstrings.dtype(), torch.int8)


def test_dtype_matches_bitstring_dtype() -> None:
    check.equal(bitstrings.dtype(), bitstring.dtype())


def test_device_matches_bitstring_device() -> None:
    check.equal(bitstrings.device(), bitstring.device())


def test_zeros_creates_int8_zero_matrix() -> None:
    m, n = 3, 5
    result = bitstrings.zeros(m, n)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (m, n))
    torch.testing.assert_close(result, torch.zeros((m, n), dtype=torch.int8))


def test_zeros_creates_single_element_zero_matrix() -> None:
    result = bitstrings.zeros(1, 1)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (1, 1))
    torch.testing.assert_close(result, torch.zeros((1, 1), dtype=torch.int8))


def test_zeros_creates_empty_matrix_when_m_is_zero() -> None:
    result = bitstrings.zeros(0, 5)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0, 5))


def test_zeros_creates_empty_matrix_when_n_is_zero() -> None:
    result = bitstrings.zeros(3, 0)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (3, 0))


def test_zeros_creates_zero_matrix_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    m, n = 2, 4
    result = bitstrings.zeros(m, n, device=custom_device)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (m, n))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.zeros((m, n), dtype=torch.int8, device=custom_device))


def test_tensor_creates_int8_tensor_from_2d_list() -> None:
    data = [[0, 1, 1], [1, 0, 1]]
    result = bitstrings.tensor(data)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (2, 3))
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int8))


def test_tensor_creates_int8_tensor_from_empty_list() -> None:
    result = bitstrings.tensor([])
    check.equal(result.dtype, torch.int8)
    check.equal(len(result), 0)


def test_tensor_creates_int8_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [[1, 0], [0, 1], [1, 1]]
    result = bitstrings.tensor(data, device=custom_device)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (3, 2))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int8, device=custom_device))


def test_as_tensor_converts_dtype_and_device() -> None:
    source = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    result = bitstrings.as_tensor(source)
    check.equal(result.dtype, torch.int8)
    check.equal(result.device, bitstrings.device())
    torch.testing.assert_close(result, torch.tensor([[1, 0], [0, 1]], dtype=torch.int8))


def test_as_tensor_preserves_values() -> None:
    source = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float32)
    result = bitstrings.as_tensor(source)
    check.equal(result.dtype, torch.int8)
    torch.testing.assert_close(result, torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int8))


def test_from_strings_creates_tensor_from_single_string() -> None:
    result = bitstrings.from_strings(["101"])
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (1, 3))
    torch.testing.assert_close(result, torch.tensor([[1, 0, 1]], dtype=torch.int8))


def test_from_strings_creates_tensor_from_multiple_strings() -> None:
    result = bitstrings.from_strings(["011", "101", "000"])
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (3, 3))
    torch.testing.assert_close(
        result, torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=torch.int8)
    )


def test_from_strings_creates_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    result = bitstrings.from_strings(["10", "01"], device=custom_device)
    check.equal(result.device, custom_device)


def test_from_strings_creates_all_zeros_tensor() -> None:
    result = bitstrings.from_strings(["0000", "0000", "0000"])
    check.equal(result.shape, (3, 4))
    torch.testing.assert_close(result, torch.zeros((3, 4), dtype=torch.int8))


def test_from_strings_creates_all_ones_tensor() -> None:
    result = bitstrings.from_strings(["111", "111"])
    torch.testing.assert_close(result, torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int8))


def test_from_strings_roundtrips_with_to_strings() -> None:
    original = ["01101", "10010", "11111", "00000"]
    roundtripped = bitstrings.to_strings(bitstrings.from_strings(original))
    check.equal(roundtripped, original)


def test_from_strings_raises_on_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        bitstrings.from_strings(["01", "101", "0"])


def test_from_strings_returns_empty_tensor_for_empty_list() -> None:
    result = bitstrings.from_strings([])
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0, 0))


@pytest.mark.parametrize("seed", [73, 256, 8401])
@pytest.mark.parametrize("count", [1, 2, 5, 10])
@pytest.mark.parametrize("length", [1, 2, 5, 10, 50])
def test_from_strings_roundtrips_with_to_strings_random(seed: int, count: int, length: int) -> None:
    rng = random.Random(seed)
    original = ["".join(rng.choice("01") for _ in range(length)) for _ in range(count)]
    roundtripped = bitstrings.to_strings(bitstrings.from_strings(original))
    check.equal(roundtripped, original)


def test_to_strings_returns_empty_list_for_empty_tensor() -> None:
    empty = bitstrings.zeros(0, 5)
    result = bitstrings.to_strings(empty)
    check.equal(result, [])


def test_to_strings_returns_single_string_for_single_row() -> None:
    bs = bitstrings.tensor([[1, 0, 1]])
    result = bitstrings.to_strings(bs)
    check.equal(result, ["101"])


def test_to_strings_returns_correct_strings_for_multiple_rows() -> None:
    bs = bitstrings.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    result = bitstrings.to_strings(bs)
    check.equal(result, ["011", "101", "000"])


def test_to_strings_returns_all_zeros_strings() -> None:
    bs = bitstrings.zeros(3, 4)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["0000", "0000", "0000"])


def test_to_strings_returns_all_ones_strings() -> None:
    bs = bitstrings.tensor([[1, 1, 1], [1, 1, 1]])
    result = bitstrings.to_strings(bs)
    check.equal(result, ["111", "111"])


def test_to_strings_length_matches_number_of_rows() -> None:
    bs = bitstrings.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
    result = bitstrings.to_strings(bs)
    check.equal(len(result), 4)


def test_to_strings_as_tensor_converts_correctly() -> None:
    source = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int32)
    bs = bitstrings.as_tensor(source)
    result = bitstrings.to_strings(bs)
    check.equal(result, ["101", "010"])


def test_rand_creates_int8_matrix_of_requested_shape() -> None:
    count, n_bits = 6, 20
    result = bitstrings.rand(count, n_bits, rng=torch_rng(0))
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (count, n_bits))


def test_rand_creates_empty_matrix_when_count_is_zero() -> None:
    result = bitstrings.rand(0, 5, rng=torch_rng(0))
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0, 5))


def test_rand_creates_empty_matrix_when_n_bits_is_zero() -> None:
    result = bitstrings.rand(5, 0, rng=torch_rng(0))
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (5, 0))


def test_rand_only_contains_zeros_and_ones() -> None:
    result = bitstrings.rand(20, 20, rng=torch_rng(0))
    check.is_true(torch.all((result == 0) | (result == 1)).item())


def test_rand_creates_matrix_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    result = bitstrings.rand(4, 10, device=custom_device, rng=torch_rng(0))
    check.equal(result.device, custom_device)


def test_rand_is_deterministic_with_seeded_rng() -> None:
    a = bitstrings.rand(10, 50, rng=torch_rng(1234))
    b = bitstrings.rand(10, 50, rng=torch_rng(1234))
    torch.testing.assert_close(a, b)


def test_rand_differs_across_seeds() -> None:
    a = bitstrings.rand(10, 50, rng=torch_rng(1))
    b = bitstrings.rand(10, 50, rng=torch_rng(2))
    check.is_false(torch.equal(a, b))


def test_rand_rows_are_independent_not_identical() -> None:
    result = bitstrings.rand(10, 30, rng=torch_rng(0))
    check.is_false(torch.all(result == result[0]).item())


def test_as_tensor_creates_int8_tensor_from_list() -> None:
    data = [[1, 0, 1], [1, 0, 0]]
    result = bitstrings.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstrings.dtype(), device=bitstrings.device())
    )


def test_as_tensor_creates_int8_tensor_from_numpy_array() -> None:
    data = np.array([[0, 1, 1], [0, 1, 0]])
    result = bitstrings.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstrings.dtype(), device=bitstrings.device())
    )


def test_as_tensor_no_copy_when_input_already_matches_dtype_and_device() -> None:
    source = torch.tensor([[1, 0], [1, 0]], dtype=bitstrings.dtype(), device=bitstrings.device())
    result = bitstrings.as_tensor(source)
    check.is_(result, source)
    source[0][0] = 0
    check.equal(result[0][0].item(), 0)


def test_as_tensor_copies_when_dtype_differs() -> None:
    source = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
    result = bitstrings.as_tensor(source)
    check.equal(result.dtype, bitstrings.dtype())
    check.is_not(result, source)
    source[0][0] = 5
    check.equal(result[0][0].item(), 1)


def test_as_tensor_no_copy_when_numpy_array_already_matches_dtype_on_cpu() -> None:
    source = np.array([[1, 0], [1, 0]], dtype=np.int8)
    result = bitstrings.as_tensor(source)
    check.equal(result.dtype, bitstrings.dtype())
    source[0][0] = 5
    check.equal(result[0][0].item(), 5)


def test_as_tensor_copies_when_numpy_array_dtype_differs() -> None:
    source = np.array([[1, 0], [1, 0]], dtype=np.int32)
    result = bitstrings.as_tensor(source)
    check.equal(result.dtype, bitstrings.dtype())
    source[0][0] = 5
    check.equal(result[0][0].item(), 1)


def test_as_tensor_copies_when_input_is_a_list() -> None:
    data = [[1, 0], [1, 0]]
    result = bitstrings.as_tensor(data)
    check.equal(result.dtype, bitstrings.dtype())
    data[0][0] = 5
    check.equal(result[0][0].item(), 1)


def test_as_tensor_preserves_values() -> None:
    data = [[0, 1, 1], [0, 1, 0]]
    result = bitstrings.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstrings.dtype(), device=bitstrings.device())
    )
