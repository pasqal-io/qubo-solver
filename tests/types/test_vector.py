from __future__ import annotations

import torch
import pytest_check as check

from qubosolver import vector, linalg


def test_dtype_returns_linalg_dtype() -> None:
    check.equal(vector.dtype(), linalg.dtype())


def test_device_returns_linalg_device() -> None:
    check.equal(vector.device(), linalg.device())


def test_zeros_creates_float_zero_vector() -> None:
    n = 5
    result = vector.zeros(n)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (n,))
    torch.testing.assert_close(result, torch.zeros(n, dtype=linalg.dtype()))


def test_zeros_creates_single_element_zero_vector() -> None:
    result = vector.zeros(1)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (1,))
    torch.testing.assert_close(result, torch.zeros(1, dtype=linalg.dtype()))


def test_zeros_creates_empty_vector_when_n_is_zero() -> None:
    result = vector.zeros(0)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_zeros_creates_zero_vector_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    n = 4
    result = vector.zeros(n, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (n,))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(result, torch.zeros(n, dtype=linalg.dtype(), device=custom_device))


def test_zeros_with_custom_dtype() -> None:
    n = 3
    result = vector.zeros(n, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    check.equal(result.shape, (n,))
    torch.testing.assert_close(result, torch.zeros(n, dtype=torch.float64))


def test_tensor_creates_float_tensor_from_list() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = vector.tensor(data)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (len(data),))
    torch.testing.assert_close(result, torch.tensor(data, dtype=linalg.dtype()))


def test_tensor_creates_tensor_from_empty_list() -> None:
    result = vector.tensor([])
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (0,))
    check.equal(len(result), 0)


def test_tensor_creates_tensor_on_specified_device() -> None:
    custom_device = torch.device("cpu")
    data = [1.5, 2.5, 3.5]
    result = vector.tensor(data, device=custom_device)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.shape, (len(data),))
    check.equal(result.device, custom_device)
    torch.testing.assert_close(
        result, torch.tensor(data, dtype=linalg.dtype(), device=custom_device)
    )


def test_tensor_with_custom_dtype() -> None:
    data = [1.0, 2.0, 3.0]
    result = vector.tensor(data, dtype=torch.float64)
    check.equal(result.dtype, torch.float64)
    torch.testing.assert_close(result, torch.tensor(data, dtype=torch.float64))


def test_from_torch_converts_dtype_and_device() -> None:
    source = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = vector.from_torch(source)
    check.equal(result.dtype, linalg.dtype())
    check.equal(result.device, linalg.device())
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0], dtype=linalg.dtype()))


def test_from_torch_preserves_values() -> None:
    source = torch.tensor([4, 5, 6], dtype=torch.int32)
    result = vector.from_torch(source)
    check.equal(result.dtype, linalg.dtype())
    torch.testing.assert_close(result, torch.tensor([4.0, 5.0, 6.0], dtype=linalg.dtype()))
