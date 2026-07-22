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
        return list(map(lambda c: c.name, cls))

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


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
    BLADE = "blade"


class DriveType(Enum):
    """
    Type of drive shaping method used for solving the QUBO
    """

    OPTIMIZED = "optimized"
    HEURISTIC = "heuristic"


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
    TRIVIALDIAGONAL = "trivial-diagonal"


class ClassicalSolverType(StrEnum):
    """Type of classical solver used."""

    SIMULATED_ANNEALING_TABU_SEARCH = "simulated_annealing_tabu_search"
    TABU_SEARCH = "tabu_search"
    SIMULATED_ANNEALING = "simulated_annealing"
    CPLEX = "cplex"
    RANDOM = "random"
