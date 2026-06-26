"""Embedding methods for mapping QUBO variables onto quantum hardware registers.

Provides BLaDE and greedy embedding strategies.
"""

from __future__ import annotations

from qubosolver.embedding import blade, greedy

__all__ = [
    "blade",
    "greedy",
]
