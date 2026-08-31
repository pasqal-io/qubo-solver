"""Embedding algorithms for mapping QUBO variables onto quantum hardware registers.
"""

from __future__ import annotations

from qubosolver.embedding import blade, greedy_layout
from qubosolver.embedding.enums import Lattice

__all__ = [
    "blade",
    "greedy_layout",
    "Lattice",
]
