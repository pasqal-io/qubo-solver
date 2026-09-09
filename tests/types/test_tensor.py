from __future__ import annotations

import numpy as np
import torch
import pytest_check as check

from qubosolver import tensor, linalg


def test_dtype_returns_linalg_dtype() -> None:
    check.equal(tensor.dtype(), linalg.dtype())


def test_device_returns_linalg_device() -> None:
    check.equal(tensor.device(), linalg.device())


def test_zeros_creates_float_zero_1d() -> None:
    n = 5
    result = tensor.zeros(n)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (n,))
    torch.testing.assert_close(result, torch.zeros(n, dtype=linalg.dtype()))


def test_zeros_creates_float_zero_2d() -> None:
    result = tensor.zeros(3, 4)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (3, 4))
    torch.testing.assert_close(result, torch.zeros((3, 4), dtype=linalg.dtype()))


def test_zeros_creates_float_zero_3d() -> None:
    result = tensor.zeros(2, 3, 4)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (2, 3, 4))
    torch.testing.assert_close(result, torch.zeros((2, 3, 4), dtype=linalg.dtype()))


def test_zeros_creates_single_element() -> None:
    result = tensor.zeros(1)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (1,))
    torch.testing.assert_close(result, torch.zeros(1, dtype=linalg.dtype()))


def test_zeros_creates_empty_when_zero_size() -> None:
    result = tensor.zeros(0)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_zeros_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    result = tensor.zeros(3, 4, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (3, 4))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(
        result, torch.zeros((3, 4), dtype=linalg.dtype(), device=custom_device)
    )


def test_zeros_with_custom_dtype() -> None:
    result = tensor.zeros(2, 3, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    check.equal(result.shape, (2, 3))
    torch.testing.assert_close(result, torch.zeros((2, 3), dtype=torch.float64))


def test_tensor_creates_float_tensor_from_1d_list() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = tensor.tensor(data)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (len(data),))
    torch.testing.assert_close(result, torch.tensor(data, dtype=linalg.dtype()))


def test_tensor_creates_float_tensor_from_2d_list() -> None:
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = tensor.tensor(data)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (2, 2))
    torch.testing.assert_close(result, torch.tensor(data, dtype=linalg.dtype()))


def test_tensor_creates_tensor_from_empty_list() -> None:
    result = tensor.tensor([])
    check.equal(result.dtype, linalg.dtype())
    check.equal(len(result), 0)


def test_tensor_creates_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [1.5, 2.5, 3.5]
    result = tensor.tensor(data, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (len(data),))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=linalg.dtype(), device=custom_device)
    )


def test_tensor_with_custom_dtype() -> None:
    data = [1.0, 2.0, 3.0]
    result = tensor.tensor(data, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.float64))


def test_as_tensor_creates_float_tensor_from_list() -> None:
    data = [1.0, 2.0, 3.0]
    result = tensor.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=tensor.dtype(), device=tensor.device())
    )


def test_as_tensor_creates_float_tensor_from_numpy_array() -> None:
    data = np.array([1.0, 2.0, 3.0])
    result = tensor.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=tensor.dtype(), device=tensor.device())
    )


def test_as_tensor_no_copy_when_input_already_matches_dtype_and_device() -> None:
    source = torch.tensor([1.0, 2.0, 3.0], dtype=tensor.dtype(), device=tensor.device())
    result = tensor.as_tensor(source)
    check.is_(result, source)
    source[0] = 0.0
    check.equal(result[0].item(), 0.0)


def test_as_tensor_copies_when_dtype_differs() -> None:
    other_dtype = torch.float64 if tensor.dtype() == torch.float32 else torch.float32
    source = torch.tensor([1.0, 2.0, 3.0], dtype=other_dtype)
    result = tensor.as_tensor(source)
    check.equal(result.dtype, tensor.dtype())
    check.is_not(result, source)
    source[0] = 5.0
    check.equal(result[0].item(), 1.0)


def test_as_tensor_no_copy_when_numpy_array_already_matches_dtype_on_cpu() -> None:
    np_dtype = np.float32 if tensor.dtype() == torch.float32 else np.float64
    source = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
    result = tensor.as_tensor(source)
    check.equal(result.dtype, tensor.dtype())
    source[0] = 5.0
    check.equal(result[0].item(), 5.0)


def test_as_tensor_copies_when_numpy_array_dtype_differs() -> None:
    other_np_dtype = np.float64 if tensor.dtype() == torch.float32 else np.float32
    source = np.array([1.0, 2.0, 3.0], dtype=other_np_dtype)
    result = tensor.as_tensor(source)
    check.equal(result.dtype, tensor.dtype())
    source[0] = 5.0
    check.equal(result[0].item(), 1.0)


def test_as_tensor_copies_when_input_is_a_list() -> None:
    data = [1.0, 2.0, 3.0]
    result = tensor.as_tensor(data)
    check.equal(result.dtype, tensor.dtype())
    data[0] = 5.0
    check.equal(result[0].item(), 1.0)


def test_as_tensor_preserves_values() -> None:
    data = [1.0, 2.0, 3.0]
    result = tensor.as_tensor(data)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=tensor.dtype(), device=tensor.device())
    )
