"""Enum types used by the embedding module."""

from __future__ import annotations

from enum import Enum

from pulser.register.special_layouts import SquareLatticeLayout, TriangularLatticeLayout

from qubosolver.types.enums import _StrEnum


class Algorithm(_StrEnum):
    """Type of embedding algorithm used to map the QUBO graph onto a hardware register."""

    GREEDY = "greedy"
    """Greedy layout-based embedder that places qubits on a regular lattice."""
    BLADE = "blade"
    """BLADE embedder using graph-theoretic optimization for qubit placement."""


class Layout(Enum):
    """Type of lattice layout used by the greedy embedding method."""

    SQUARE = SquareLatticeLayout
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = TriangularLatticeLayout
    """Arrange qubits on a triangular lattice grid."""
