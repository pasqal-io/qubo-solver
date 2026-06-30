from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
import pytest_check as check

from qubosolver import linalg


def test_device_from_env_defaults_to_cpu() -> None:
    with patch.dict(os.environ, {}, clear=True):
        result = linalg._device_from_env()
    check.equal(result, torch.device("cpu"))


def test_device_from_env_use_gpu_1() -> None:
    with patch.dict(os.environ, {"USE_GPU": "1"}, clear=False):
        result = linalg._device_from_env()
    check.equal(result, torch.device("cuda"))


def test_device_from_env_use_gpu_false() -> None:
    with patch.dict(os.environ, {"USE_GPU": "false"}, clear=False):
        result = linalg._device_from_env()
    check.equal(result, torch.device("cpu"))


def test_device_from_env_invalid_qubo_solver_device_raises() -> None:
    with patch.dict(os.environ, {"QUBO_SOLVER_DEVICE": "invalid_device_string"}, clear=False):
        with pytest.raises(ValueError, match="Invalid QUBO_SOLVER_DEVICE"):
            linalg._device_from_env()


def test_float_type_from_env_float64_dtype() -> None:
    with patch.dict(os.environ, {"QUBO_SOLVER_FLOAT_DTYPE": "float64"}, clear=False):
        result = linalg._float_type_from_env()
    check.equal(result, torch.float64)


def test_float_type_from_env_defaults_to_float32() -> None:
    with patch.dict(os.environ, {}, clear=True):
        result = linalg._float_type_from_env()
    check.equal(result, torch.float32)


def test_float_type_from_env_invalid_dtype_raises() -> None:
    with patch.dict(os.environ, {"QUBO_SOLVER_FLOAT_DTYPE": "float128"}, clear=False):
        with pytest.raises(ValueError, match="Invalid QUBO_SOLVER_FLOAT_DTYPE"):
            linalg._float_type_from_env()


def test_global_config_use_double_precision_sets_float64() -> None:
    original = linalg._GlobalConfig._float_dtype
    try:
        linalg._GlobalConfig.use_double_precision(enable=True)
        check.equal(linalg._GlobalConfig._float_dtype, torch.float64)
    finally:
        linalg._GlobalConfig._float_dtype = original


def test_global_config_set_float_precision_invalid_dtype_raises() -> None:
    with pytest.raises(ValueError):
        linalg._GlobalConfig.set_float_precision(torch.int32)


def test_global_config_use_gpu_sets_cuda_device() -> None:
    original = linalg._GlobalConfig._device
    try:
        linalg._GlobalConfig.use_gpu(enable=True)
        check.equal(linalg._GlobalConfig._device, torch.device("cuda"))
        check.equal(linalg.device(), torch.device("cuda"))
    finally:
        linalg._GlobalConfig._device = original
