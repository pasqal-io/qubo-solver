from __future__ import annotations

import os
import torch
from ._checks import TYPE_CHECKING

_FLOAT_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _device_from_env() -> torch.device:
    """Returns the torch device based on environment variables.

    Checks in order:
    1. QUBO_SOLVER_DEVICE — explicit device string (e.g. "cpu", "cuda", "cuda:1", "mps")
    2. USE_GPU — if set to "1" or "true", use "cuda"; otherwise "cpu"
    3. Defaults to "cpu" if neither is set.
    """
    device_str = os.getenv("QUBO_SOLVER_DEVICE")
    if device_str is not None:
        try:
            return torch.device(device_str)
        except RuntimeError:
            raise ValueError(f"Invalid QUBO_SOLVER_DEVICE={device_str!r}.")

    use_gpu = os.getenv("USE_GPU")
    if use_gpu is not None:
        return torch.device("cuda" if use_gpu.lower() in ("1", "true") else "cpu")

    return torch.device("cpu")


def _use_double_precision_from_env() -> bool:
    """Returns whether double precision (float64) is enabled based on environment variables.

    Checks in order:
    1. QUBO_SOLVER_FLOAT_DTYPE — if set to "float64", returns True; "float32" returns False
    2. USE_DOUBLE_PRECISION — if set to "1" or "true", returns True; otherwise False
    3. Defaults to False if neither is set (i.e. float32 is the default, matching PyTorch).
    """
    float_dtype_str = os.getenv("QUBO_SOLVER_FLOAT_DTYPE")
    if float_dtype_str is not None:
        return float_dtype_str == "float64"

    use_double = os.getenv("USE_DOUBLE_PRECISION")
    if use_double is not None:
        return use_double.lower() in ("1", "true")

    return False


def _float_type_from_env() -> torch.dtype:
    """Returns the float dtype based on environment variables.

    Checks in order:
    1. QUBO_SOLVER_FLOAT_DTYPE — explicit dtype name (float16, bfloat16, float32, float64)
    2. USE_DOUBLE_PRECISION — if set to "1" or "true", use float64; otherwise float32
    3. Defaults to float32 if neither is set (matching PyTorch default).
    """
    float_dtype_str = os.getenv("QUBO_SOLVER_FLOAT_DTYPE")
    if float_dtype_str is not None:
        if float_dtype_str not in _FLOAT_DTYPE_MAP:
            raise ValueError(
                f"Invalid QUBO_SOLVER_FLOAT_DTYPE={float_dtype_str!r}. "
                f"Valid options: {list(_FLOAT_DTYPE_MAP)}"
            )
        return _FLOAT_DTYPE_MAP[float_dtype_str]

    return torch.float64 if _use_double_precision_from_env() else torch.float32


class _GlobalConfig:
    _float_dtype = _float_type_from_env()
    _device = _device_from_env()

    @classmethod
    def use_double_precision(cls, enable: bool = True) -> None:
        """Allows the user to easily toggle float64 on or off."""
        cls._float_dtype = torch.float64 if enable else torch.float32

    @classmethod
    def set_float_precision(cls, dtype: torch.dtype) -> None:
        """Set the global float dtype.

        Args:
            dtype: A torch float dtype. Valid options: float16, bfloat16, float32, float64.
        """
        if dtype not in _FLOAT_DTYPE_MAP.values():
            raise ValueError(
                f"Invalid dtype {dtype!r}. Valid options: {list(_FLOAT_DTYPE_MAP.values())}"
            )
        cls._float_dtype = dtype

    @classmethod
    def use_gpu(cls, enable: bool = True) -> None:
        """Allows the user to easily toggle GPU (cuda) on or off."""
        cls._device = torch.device("cuda" if enable else "cpu")

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the global torch device.

        Args:
            device: A torch.device instance (e.g. torch.device("cuda:1")).
        """
        cls._device = device


def dtype() -> torch.dtype:
    """Returns the currently configured global float dtype."""
    return _GlobalConfig._float_dtype


def device() -> torch.device:
    """Returns the currently configured global torch device."""
    return _GlobalConfig._device


if TYPE_CHECKING:
    import jaxtyping
    from typing import Final
    from typing_extensions import TypeAlias

    _USE_DOUBLE_PRECISION: Final[bool] = _use_double_precision_from_env()

    Vectorf: TypeAlias = jaxtyping.Float32[torch.Tensor, "n"]  # noqa: F821
    Matrixf: TypeAlias = jaxtyping.Float32[torch.Tensor, "n n"]  # noqa: F821, F722
    Tensorf: TypeAlias = jaxtyping.Float32[torch.Tensor, "..."]  # noqa: F821

    Vectord: TypeAlias = jaxtyping.Float64[torch.Tensor, "n"]  # noqa: F821
    Matrixd: TypeAlias = jaxtyping.Float64[torch.Tensor, "n n"]  # noqa: F821, F722
    Tensord: TypeAlias = jaxtyping.Float64[torch.Tensor, "..."]  # noqa: F821

    Vectori: TypeAlias = jaxtyping.Int64[torch.Tensor, "n"]  # noqa: F821

    if _USE_DOUBLE_PRECISION:
        Vector = Vectord
        Matrix = Matrixd
        Tensor = Tensord
    else:
        Vector = Vectorf
        Matrix = Matrixf
        Tensor = Tensorf

    Bitstring: TypeAlias = jaxtyping.Int8[torch.Tensor, "n"]  # noqa: F821
    Bitstrings: TypeAlias = jaxtyping.Int8[torch.Tensor, "n m"]  # noqa: F821, F722

else:
    Vectorf: TypeAlias = torch.Tensor
    Matrixf: TypeAlias = torch.Tensor
    Tensorf: TypeAlias = torch.Tensor

    Vectord: TypeAlias = torch.Tensor
    Matrixd: TypeAlias = torch.Tensor
    Tensord: TypeAlias = torch.Tensor

    Vectori: TypeAlias = torch.Tensor

    Vector: TypeAlias = torch.Tensor
    Matrix: TypeAlias = torch.Tensor
    Tensor: TypeAlias = torch.Tensor

    Bitstring: TypeAlias = torch.Tensor
    Bitstrings: TypeAlias = torch.Tensor
