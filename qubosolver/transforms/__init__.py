"""Transforms for QUBO instances.

Provides preprocessing transforms such as variable fixing to reduce
problem size before solving.
"""

from __future__ import annotations

from qubosolver.transforms import variable_fixing

__all__ = [
    # Submodules
    "variable_fixing",
]
