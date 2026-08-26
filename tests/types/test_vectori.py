from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
import pytest_check as check

from qubosolver import vectori, linalg


def test_dtype_returns_int64() -> None:
    check.equal(vectori.dtype(), torch.int64)


def test_device_returns_linalg_device() -> None:
    check.equal(vectori.device(), linalg.device())


def test_zeros_creates_int64_zero_vector() -> None:
    n = 5
    result = vectori.zeros(n)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (n,))
    torch.testing.assert_close(result, torch.zeros(n, dtype=torch.int64))


def test_zeros_creates_single_element_zero_vector() -> None:
    result = vectori.zeros(1)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (1,))
    torch.testing.assert_close(result, torch.zeros(1, dtype=torch.int64))


def test_zeros_creates_empty_vector_when_n_is_zero() -> None:
    result = vectori.zeros(0)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_zeros_creates_zero_vector_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    n = 4
    result = vectori.zeros(n, device=custom_device)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (n,))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.zeros(n, dtype=torch.int64, device=custom_device))


def test_tensor_creates_int64_tensor_from_list() -> None:
    data = [10, 20, 30, 40, 50]
    result = vectori.tensor(data)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (len(data),))
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int64))


def test_tensor_creates_tensor_from_empty_list() -> None:
    result = vectori.tensor([])
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_tensor_creates_int64_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [1, 2, 3, 4]
    result = vectori.tensor(data, device=custom_device)
    check.equal(result.dtype, torch.int64)
    check.equal(result.shape, (len(data),))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.int64, device=custom_device))


def test_tensor_propagates_kwargs_to_vector_tensor() -> None:
    data = [1, 2, 3]
    with patch("qubosolver.types.vectori.vector.tensor") as mock_vector_tensor:
        mock_vector_tensor.return_value = torch.tensor(data, dtype=torch.int64)
        vectori.tensor(data, requires_grad=False)
        mock_vector_tensor.assert_called_once()
        call_kwargs = mock_vector_tensor.call_args
        check.is_false(call_kwargs.kwargs.get("requires_grad"))


def test_as_tensor_converts_dtype_and_device() -> None:
    source = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = vectori.as_tensor(source)
    check.equal(result.dtype, torch.int64)
    check.equal(result.device, linalg.device())
    torch.testing.assert_close(result, torch.tensor([1, 2, 3], dtype=torch.int64))


def test_as_tensor_preserves_values() -> None:
    source = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    result = vectori.as_tensor(source)
    check.equal(result.dtype, torch.int64)
    torch.testing.assert_close(result, torch.tensor([4, 5, 6], dtype=torch.int64))


def test_as_tensor_creates_int64_tensor_from_list() -> None:
    data = [1, 2, 3, 4, 5]
    result = vectori.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=vectori.dtype(), device=vectori.device())
    )


def test_as_tensor_creates_int64_tensor_from_numpy_array() -> None:
    data = np.array([1, 2, 3, 4, 5])
    result = vectori.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=vectori.dtype(), device=vectori.device())
    )


def test_as_tensor_no_copy_when_input_already_matches_dtype_and_device() -> None:
    source = torch.tensor([1, 2, 3, 4], dtype=vectori.dtype(), device=vectori.device())
    result = vectori.as_tensor(source)
    check.is_(result, source)
    source[0] = 0
    check.equal(result[0].item(), 0)


def test_as_tensor_copies_when_dtype_differs() -> None:
    source = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    result = vectori.as_tensor(source)
    check.equal(result.dtype, vectori.dtype())
    check.is_not(result, source)
    source[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_no_copy_when_numpy_array_already_matches_dtype_on_cpu() -> None:
    source = np.array([1, 2, 3, 4], dtype=np.int64)
    result = vectori.as_tensor(source)
    check.equal(result.dtype, vectori.dtype())
    source[0] = 5
    check.equal(result[0].item(), 5)


def test_as_tensor_copies_when_numpy_array_dtype_differs() -> None:
    source = np.array([1, 2, 3, 4], dtype=np.int32)
    result = vectori.as_tensor(source)
    check.equal(result.dtype, vectori.dtype())
    source[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_copies_when_input_is_a_list() -> None:
    data = [1, 2, 3, 4]
    result = vectori.as_tensor(data)
    check.equal(result.dtype, vectori.dtype())
    data[0] = 5
    check.equal(result[0].item(), 1)


def test_as_tensor_preserves_values() -> None:
    data = [1, 2, 3, 4, 5]
    result = vectori.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=vectori.dtype(), device=vectori.device())
    )
