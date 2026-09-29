"""Embedding algorithms for mapping QUBO variables onto quantum hardware registers."""

from __future__ import annotations

from qubosolver.embedding import blade, greedy_layout
from qubosolver.embedding.enums import Lattice, Norm

__all__ = [
    "Lattice",
    "Norm",
    "blade",
    "greedy_layout",
]
