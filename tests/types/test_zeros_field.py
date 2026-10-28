# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from dataclasses import dataclass

import pytest_check as check
import torch

from qubosolver import Bitstring, Matrix, Tensor, bitstring, linalg, matrix, tensor


@dataclass
class _WithZeros:
    b: Bitstring = bitstring.zeros_field(3)  # noqa: RUF009
    m: Matrix = matrix.zeros_field(2)  # noqa: RUF009
    t: Tensor = tensor.zeros_field(2, 3, dtype=torch.int64)  # noqa: RUF009


def test_zeros_field_defaults() -> None:
    first = _WithZeros()
    torch.testing.assert_close(first.b, bitstring.zeros(3))
    torch.testing.assert_close(first.m, torch.zeros(2, 2, dtype=linalg.dtype()))
    torch.testing.assert_close(first.t, torch.zeros(2, 3, dtype=torch.int64))


def test_zeros_field_builds_a_fresh_tensor_per_instance() -> None:
    first, second = _WithZeros(), _WithZeros()
    check.is_not(first.b, second.b)
    check.is_not(first.m, second.m)
    check.is_not(first.t, second.t)
