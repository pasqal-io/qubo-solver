# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Hybrid quantum-classical solvers combining Pasqal devices with classical optimization."""

from __future__ import annotations

from qubosolver.solving.hybrid import (
    drive_bayesian_search,
)

__all__ = [
    "drive_bayesian_search",
]
