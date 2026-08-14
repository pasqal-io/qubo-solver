"""Embedding methods for mapping QUBO variables onto quantum hardware registers.

Provides BLaDE and greedy embedding strategies.
"""

from __future__ import annotations

from qubosolver.embedding import blade, greedy_layout
from qubosolver.embedding.enum import Algorithm, Layout

__all__ = [
    "blade",
    "greedy_layout",
    "Algorithm",
    "Layout",
]
