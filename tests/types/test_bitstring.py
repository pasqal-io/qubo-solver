from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
import pytest_check as check

from qubosolver import bitstring, linalg, torch_rng


def test_dtype_returns_int8() -> None:
    check.equal(bitstring.dtype(), torch.int8)


def test_device_returns_linalg_device() -> None:
    check.equal(bitstring.device(), linalg.device())


def test_zeros_creates_int8_zero_vector() -> None:
    n = 5
    result = bitstring.zeros(n)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (n,))
    torch.testing.assert_close(result, torch.zeros(n, dtype=torch.int8))


def test_zeros_creates_single_element_zero_vector() -> None:
    result = bitstring.zeros(1)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (1,))
    torch.testing.assert_close(result, torch.zeros(1, dtype=torch.int8))


def test_zeros_creates_empty_vector_when_n_is_zero() -> None:
    result = bitstring.zeros(0)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_zeros_creates_zero_vector_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    n = 4
    result = bitstring.zeros(n, device=custom_device)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (n,))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.zeros(n, dtype=torch.int8, device=custom_device))


def test_tensor_creates_int8_tensor_from_list() -> None:
    data = [0, 1, 1, 0, 1]
    result = bitstring.tensor(data)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (len(data),))
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int8))


def test_tensor_creates_int8_tensor_from_empty_list() -> None:
    result = bitstring.tensor([])
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_tensor_creates_int8_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [1, 0, 1, 1]
    result = bitstring.tensor(data, device=custom_device)
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (len(data),))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int8, device=custom_device))


def test_tensor_propagates_kwargs_to_vector_tensor() -> None:
    data = [1, 0, 1]
    with patch("qubosolver.types.bitstring.vector.tensor") as mock_vector_tensor:
        mock_vector_tensor.return_value = torch.tensor(data, dtype=torch.int8)
        bitstring.tensor(data, requires_grad=False)
        mock_vector_tensor.assert_called_once()
        call_kwargs = mock_vector_tensor.call_args
        check.is_false(call_kwargs.kwargs.get("requires_grad"))


def test_as_tensor_converts_dtype_and_device() -> None:
    source = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    result = bitstring.as_tensor(source)
    check.equal(result.dtype, torch.int8)
    check.equal(result.device, linalg.device())
    torch.testing.assert_close(result, torch.tensor([1, 0, 1, 0], dtype=torch.int8))


def test_as_tensor_preserves_values() -> None:
    source = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
    result = bitstring.as_tensor(source)
    check.equal(result.dtype, torch.int8)
    torch.testing.assert_close(result, torch.tensor([0, 1, 1, 0, 1], dtype=torch.int8))


def test_from_string_creates_bitstring_from_empty_string() -> None:
    result = bitstring.from_string("")
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_from_string_creates_bitstring_from_single_zero() -> None:
    result = bitstring.from_string("0")
    check.equal(result.dtype, torch.int8)
    torch.testing.assert_close(result, torch.tensor([0], dtype=torch.int8))


def test_from_string_creates_bitstring_from_single_one() -> None:
    result = bitstring.from_string("1")
    check.equal(result.dtype, torch.int8)
    torch.testing.assert_close(result, torch.tensor([1], dtype=torch.int8))


def test_from_string_creates_bitstring_from_multi_char_string() -> None:
    result = bitstring.from_string("01101")
    check.equal(result.dtype, torch.int8)
    torch.testing.assert_close(result, torch.tensor([0, 1, 1, 0, 1], dtype=torch.int8))


def test_from_string_creates_bitstring_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    result = bitstring.from_string("101", device=custom_device)
    check.equal(result.device, custom_device)


def test_from_string_roundtrips_with_to_string() -> None:
    original = "1011001011100101"
    result = bitstring.to_string(bitstring.from_string(original))
    check.equal(result, original)


@pytest.mark.parametrize("seed", [42, 137, 9991])
@pytest.mark.parametrize("length", [0, 1, 2, 5, 10, 50, 100])
def test_from_string_roundtrips_with_to_string_random(seed: int, length: int) -> None:
    rng = random.Random(seed)
    original = "".join(rng.choice("01") for _ in range(length))
    roundtripped = bitstring.to_string(bitstring.from_string(original))
    check.equal(roundtripped, original)


def test_to_string_returns_empty_string_for_empty_tensor() -> None:
    empty = bitstring.tensor([])
    result = bitstring.to_string(empty)
    check.equal(result, "")


def test_to_string_returns_zero_for_single_zero_element() -> None:
    single = bitstring.tensor([0])
    result = bitstring.to_string(single)
    check.equal(result, "0")


def test_to_string_returns_one_for_single_one_element() -> None:
    single = bitstring.tensor([1])
    result = bitstring.to_string(single)
    check.equal(result, "1")


def test_to_string_returns_correct_string_for_multi_element_tensor() -> None:
    bs = bitstring.tensor([0, 1, 1, 0, 1])
    result = bitstring.to_string(bs)
    check.equal(result, "01101")


def test_to_string_returns_all_zeros_for_zeros_bitstring() -> None:
    bs = bitstring.zeros(5)
    result = bitstring.to_string(bs)
    check.equal(result, "00000")


def test_to_string_returns_all_ones_for_ones_bitstring() -> None:
    bs = bitstring.tensor([1, 1, 1, 1, 1])
    result = bitstring.to_string(bs)
    check.equal(result, "11111")


def test_to_string_returns_correct_string_for_long_bitstring() -> None:
    bs = bitstring.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    result = bitstring.to_string(bs)
    check.equal(result, "1011001011100101")


def test_to_string_converts_bitstring_tensor_to_string() -> None:
    bs = bitstring.tensor([1, 0, 0, 1, 1])
    result = bitstring.to_string(bs)
    check.equal(result, "10011")


def test_to_string_converts_as_tensor_bitstring_to_string() -> None:
    source = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)
    bs = bitstring.as_tensor(source)
    result = bitstring.to_string(bs)
    check.equal(result, "10110")


def test_to_string_length_matches_number_of_elements() -> None:
    data = [1, 0, 1, 0, 1, 1, 0]
    bs = bitstring.tensor(data)
    result = bitstring.to_string(bs)
    check.equal(len(result), len(data))


def test_rand_creates_int8_vector_of_requested_length() -> None:
    n = 20
    result = bitstring.rand(n, rng=torch_rng(0))
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (n,))


def test_rand_creates_empty_vector_when_n_is_zero() -> None:
    result = bitstring.rand(0, rng=torch_rng(0))
    check.equal(result.dtype, torch.int8)
    check.equal(result.shape, (0,))


def test_rand_only_contains_zeros_and_ones() -> None:
    result = bitstring.rand(200, rng=torch_rng(0))
    check.is_true(torch.all((result == 0) | (result == 1)).item())


def test_rand_creates_vector_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    result = bitstring.rand(10, device=custom_device, rng=torch_rng(0))
    check.equal(result.device, custom_device)


def test_rand_is_deterministic_with_seeded_rng() -> None:
    a = bitstring.rand(50, rng=torch_rng(1234))
    b = bitstring.rand(50, rng=torch_rng(1234))
    torch.testing.assert_close(a, b)


def test_rand_differs_across_seeds() -> None:
    a = bitstring.rand(50, rng=torch_rng(1))
    b = bitstring.rand(50, rng=torch_rng(2))
    check.is_false(torch.equal(a, b))


def test_as_tensor_creates_int8_tensor_from_list() -> None:
    data = [1, 0, 1, 1, 0]
    result = bitstring.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstring.dtype(), device=bitstring.device())
    )


def test_as_tensor_creates_int8_tensor_from_numpy_array() -> None:
    data = np.array([0, 1, 1, 0, 1])
    result = bitstring.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstring.dtype(), device=bitstring.device())
    )


def test_as_tensor_no_copy_when_input_already_matches_dtype_and_device() -> None:
    source = torch.tensor([1, 0, 1, 0], dtype=bitstring.dtype(), device=bitstring.device())
    result = bitstring.as_tensor(source)
    check.is_(result, source)
    source[0] = 0
    check.equal(result[0].item(), 0)


def test_as_tensor_copies_when_dtype_differs() -> None:
    source = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    result = bitstring.as_tensor(source)
    check.equal(result.dtype, bitstring.dtype())
    check.is_not(result, source)
    source[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_no_copy_when_numpy_array_already_matches_dtype_on_cpu() -> None:
    source = np.array([1, 0, 1, 0], dtype=np.int8)
    result = bitstring.as_tensor(source)
    check.equal(result.dtype, bitstring.dtype())
    source[0] = 5
    check.equal(result[0].item(), 5)


def test_as_tensor_copies_when_numpy_array_dtype_differs() -> None:
    source = np.array([1, 0, 1, 0], dtype=np.int32)
    result = bitstring.as_tensor(source)
    check.equal(result.dtype, bitstring.dtype())
    source[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_copies_when_input_is_a_list() -> None:
    data = [1, 0, 1, 0]
    result = bitstring.as_tensor(data)
    check.equal(result.dtype, bitstring.dtype())
    data[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_preserves_values() -> None:
    data = [0, 1, 1, 0, 1]
    result = bitstring.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=bitstring.dtype(), device=bitstring.device())
    )
