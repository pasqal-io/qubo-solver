"""Enum types used by the embedding module."""

from __future__ import annotations


class Lattice():
    """Type of lattice used by the greedy embedding method."""

    SQUARE = "square"
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = "triangular"
    """Arrange qubits on a triangular lattice grid."""
