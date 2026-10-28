# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import pytest_check as check

from qubosolver.types._enums import _DensityType, _QUBOType


def test_density_type_names_and_values() -> None:
    check.equal(_DensityType.names(), ["SPARSE", "MEDIUM", "HIGH"])
    check.equal(_DensityType.list(), ["sparse", "medium", "high"])


def test_qubo_type_names_and_values() -> None:
    check.equal(_QUBOType.names(), ["MAX_CUT", "ISING_MODEL", "GENERAL_QUBO"])
    check.equal(_QUBOType.list(), ["max_cut", "ising_model", "general_qubo"])
