# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Embedding algorithms for mapping QUBO variables onto quantum hardware registers."""

from __future__ import annotations

from qubosolver.embedding import blade, greedy_layout
from qubosolver.embedding.enums import Lattice

__all__ = [
    "Lattice",
    "blade",
    "greedy_layout",
]
