from __future__ import annotations

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
