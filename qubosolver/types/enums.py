from __future__ import annotations

from enum import Enum

from pulser.register.special_layouts import SquareLatticeLayout, TriangularLatticeLayout


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


class DensityType(_StrEnum):
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


class EmbedderType(_StrEnum):
    """Type of embedding algorithm used to map the QUBO graph onto a hardware register."""

    GREEDY = "greedy"
    """Greedy layout-based embedder that places qubits on a regular lattice."""
    BLADE = "blade"
    """BLADE embedder using graph-theoretic optimization for qubit placement."""


class LayoutType(Enum):
    """Type of lattice layout used by the greedy embedding method."""

    SQUARE = SquareLatticeLayout
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = TriangularLatticeLayout
    """Arrange qubits on a triangular lattice grid."""


class DriveType(Enum):
    """Type of drive shaping method applied to the analog quantum pulse sequence."""

    OPTIMIZED = "optimized"
    """Numerically optimized drive that minimizes the cost function via pulse optimization."""
    HEURISTIC = "heuristic"
    """Fast heuristic drive based on predefined pulse shapes; no numerical optimization."""


class ClassicalSolverType(_StrEnum):
    """Type of classical solver used as a backend for QUBO optimization."""

    SIMULATED_ANNEALING_TABU_SEARCH = "simulated_annealing_tabu_search"
    """Hybrid solver combining simulated annealing with tabu-search post-processing."""
    TABU_SEARCH = "tabu_search"
    """Tabu search metaheuristic that avoids recently visited solutions."""
    SIMULATED_ANNEALING = "simulated_annealing"
    """Simulated annealing algorithm that probabilistically accepts worse solutions to escape local minima."""
    CPLEX = "cplex"
    """IBM CPLEX exact solver; requires a valid CPLEX installation and licence."""
    RANDOM = "random"
    """Randomly samples solutions; useful as a baseline or for testing."""
