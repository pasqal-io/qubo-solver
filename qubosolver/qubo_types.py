from __future__ import annotations

from enum import Enum

from pulser.register.special_layouts import SquareLatticeLayout, TriangularLatticeLayout


class StrEnum(str, Enum):
    """String-based Enums class implementation"""

    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def names(cls) -> list[str]:
        return list(map(lambda c: c.name, cls))  # type: ignore

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class DensityType(StrEnum):
    """String-based Enums for classifying density."""

    SPARSE = "sparse"
    MEDIUM = "medium"
    HIGH = "high"


class QUBOType(StrEnum):
    """
    String-based Enums for categorizing different types of
    QUBO problems.
    """

    MAX_CUT = "max_cut"
    ISING_MODEL = "ising_model"
    GENERAL_QUBO = "general_qubo"


class EmbedderType(StrEnum):
    """
    Type of embedding method to use
    """

    GREEDY = "greedy"


class LayoutType(Enum):
    """
    Type of layout for the greedy embedding method
    """

    SQUARE = SquareLatticeLayout
    TRIANGULAR = TriangularLatticeLayout


class PulseType(Enum):
    """
    Type of pulse shaping method used for solving the QUBO
    """

    ADIABATIC = "adiabatic"
    OPTIMIZED = "optimized"


class SolutionStatusType(StrEnum):
    """
    Type of solution status used for pre-post processing and trivial solution.
    """

    UNPROCESSED = "unprocessed"
    PREPROCESSED = "preprocessed"
    POSTPROCESSED = "postprocessed"
    PREPOSTPROCESSED = "pre-postprocessed"

    TRIVIALONE = "trivial-one"
    TRIVIALZERO = "trivial-zero"
