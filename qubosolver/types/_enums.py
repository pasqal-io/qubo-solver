# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Enum types used throughout the QUBO solver pipeline.

This module defines all enumeration classes that control solver behaviour —
embedding strategy and QUBO matrix density.
"""

from __future__ import annotations

from enum import StrEnum

from qubosolver._checks import TYPE_CHECKING

if TYPE_CHECKING:
    import builtins


class _StrEnum(StrEnum):
    """String-based Enums class implementation."""

    @classmethod
    def names(cls) -> builtins.list[str]:
        return [c.name for c in cls]

    @classmethod
    def list(cls) -> builtins.list[str]:  # `list` is shadowed by this method
        return [c.value for c in cls]


class _DensityType(_StrEnum):
    """String-based enum for classifying the density of a QUBO matrix."""

    SPARSE = "sparse"
    """Matrix has few non-zero off-diagonal entries relative to its size."""
    MEDIUM = "medium"
    """Matrix has a moderate number of non-zero off-diagonal entries."""
    HIGH = "high"
    """Matrix is densely connected with many non-zero off-diagonal entries."""


class _QUBOType(_StrEnum):
    """String-based enum for categorizing the structural type of a QUBO problem."""

    MAX_CUT = "max_cut"
    """QUBO encodes a Maximum Cut graph problem."""
    ISING_MODEL = "ising_model"
    """QUBO encodes an Ising model spin-glass problem."""
    GENERAL_QUBO = "general_qubo"
    """General-purpose QUBO with no specific structural pattern."""
