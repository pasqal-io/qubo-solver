"""Enum types used by the embedding module."""

from __future__ import annotations

from enum import Enum


class Lattice(Enum):
    """Type of lattice used by the [`greedy_layout`][] embedding algorithm."""

    SQUARE = "square"
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = "triangular"
    """Arrange qubits on a triangular lattice grid."""


class Norm(Enum):
    """Norm minimized by the [`greedy_layout`][] embedding algorithm.

    The objective is the deviation between the QUBO coefficients and the
    physical interactions they are embedded into, `‖U - Q‖`. The norm sets how
    individual pair deviations are combined, and therefore both where each
    qubit is placed and which starting node is retained.
    """

    L1 = "l1"
    """Sum of absolute deviations. Spreads the error evenly across pairs."""
    L2 = "l2"
    """Euclidean norm of the deviations. Penalizes large individual errors more,
    so it favours embeddings without a badly mismatched pair, possibly at the
    cost of a larger total deviation."""
