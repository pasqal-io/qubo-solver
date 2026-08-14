"""Enum types used throughout the QUBO solver pipeline.

This module defines all enumeration classes that control solver behaviour —
embedding strategy, lattice layout, drive shaping method, QUBO matrix density,
and classical solver backend.

All public enums are re-exported from the top-level `qubosolver` namespace
and can be imported directly:

```python
from qubosolver import (
    EmbedderType,
    LayoutType,
    DriveType,
    ClassicalSolverType,
)
```
"""

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

    BAYESIAN_SEARCH = "bayesian_search"
    """Drive whose parameters are found via Bayesian search that minimizes the cost function via pulse optimization."""
    PROPORTIONAL_DIAGONAL = "proportional_diagonal"
    """Drive whose amplitude/detuning scale proportionally to the QUBO diagonal; no numerical optimization."""


class ClassicalSolverType(_StrEnum):
    """Type of classical solver used as a backend for QUBO optimization."""

    TABU_SEARCH = "tabu_search"
    """Tabu search metaheuristic that avoids recently visited solutions."""
    SIMULATED_ANNEALING = "simulated_annealing"
    """Simulated annealing algorithm that probabilistically accepts worse solutions to escape local minima."""
    CPLEX = "cplex"
    """IBM CPLEX exact solver; requires a valid CPLEX installation and licence."""
    RANDOM = "random"
    """Randomly samples solutions; useful as a baseline or for testing."""
