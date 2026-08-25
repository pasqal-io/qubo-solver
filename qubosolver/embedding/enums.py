"""Enum types used by the embedding module."""

from __future__ import annotations


from qubosolver.types._enums import _StrEnum


class Algorithm(_StrEnum):
    """Type of embedding algorithm used to map the QUBO graph onto a hardware register."""

    GREEDY_LAYOUT = "greedy_layout"
    """Greedy layout-based embedder that places qubits on a regular lattice."""
    BLADE = "blade"
    """BLADE embedder using graph-theoretic optimization for qubit placement."""


class Lattice(_StrEnum):
    """Type of lattice used by the greedy embedding method."""

    SQUARE = "square"
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = "triangular"
    """Arrange qubits on a triangular lattice grid."""
