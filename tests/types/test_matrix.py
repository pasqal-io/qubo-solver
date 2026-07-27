from __future__ import annotations

import torch
import pytest_check as check

from qubosolver import matrix, linalg


def test_dtype_returns_linalg_dtype() -> None:
    check.equal(matrix.dtype(), linalg.dtype())


def test_device_returns_linalg_device() -> None:
    check.equal(matrix.device(), linalg.device())


def test_zeros_creates_float_zero_square_matrix() -> None:
    n = 5
    result = matrix.zeros(n)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (n, n))
    torch.testing.assert_close(result, torch.zeros((n, n), dtype=linalg.dtype()))


def test_zeros_creates_single_element_zero_matrix() -> None:
    result = matrix.zeros(1)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (1, 1))
    torch.testing.assert_close(result, torch.zeros((1, 1), dtype=linalg.dtype()))


def test_zeros_creates_empty_matrix_when_n_is_zero() -> None:
    result = matrix.zeros(0)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (0, 0))


def test_zeros_creates_zero_matrix_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    n = 4
    result = matrix.zeros(n, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (n, n))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(
        result, torch.zeros((n, n), dtype=linalg.dtype(), device=custom_device)
    )


def test_zeros_with_custom_dtype() -> None:
    n = 3
    result = matrix.zeros(n, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    check.equal(result.shape, (n, n))
    torch.testing.assert_close(result, torch.zeros((n, n), dtype=torch.float64))


def test_tensor_creates_float_tensor_from_2d_list() -> None:
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = matrix.tensor(data)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (3, 3))
    torch.testing.assert_close(result, torch.tensor(data, dtype=linalg.dtype()))


def test_tensor_creates_tensor_from_empty_list() -> None:
    result = matrix.tensor([])
    check.equal(result.dtype, linalg.dtype())
    check.equal(len(result), 0)


def test_tensor_creates_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = matrix.tensor(data, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (2, 2))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=linalg.dtype(), device=custom_device)
    )


def test_tensor_with_custom_dtype() -> None:
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = matrix.tensor(data, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.float64))


def test_from_torch_converts_dtype_and_device() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    result = matrix.from_torch(source)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.device, linalg.device())
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=linalg.dtype()))


def test_from_torch_preserves_values() -> None:
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    result = matrix.from_torch(source)
    check.equal(result.dtype, linalg.dtype())
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=linalg.dtype()))
