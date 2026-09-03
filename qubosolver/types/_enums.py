"""Enum types used throughout the QUBO solver pipeline.

This module defines all enumeration classes that control solver behaviour —
embedding strategy and QUBO matrix density.
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """String-based Enums class implementation"""

    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def names(cls) -> list[str]:
        return list(map(lambda c: c.name, cls))

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


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
