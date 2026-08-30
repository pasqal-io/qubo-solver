"""Enum types used by the embedding module."""

from __future__ import annotations

from enum import Enum

class Lattice(Enum):
    """Type of lattice used by the greedy embedding method."""

    SQUARE = "square"
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = "triangular"
    """Arrange qubits on a triangular lattice grid."""
