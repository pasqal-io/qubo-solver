"""Config-based entry point for building and running QUBO solvers.

Exposes [`Solver`][], the single simple entry point for building and running
QUBO solvers, along with [`SolverConfig`][] and its nested configs used to
configure it.

All names in this module are re-exported from the top-level `qubosolver`
namespace, so they can be imported directly as e.g. `from qubosolver import Solver`.
"""

from __future__ import annotations

from qubosolver.solver.config import (
    SolverConfig,
    DecompositionConfig,
    DriveShapingConfig,
    EmbeddingConfig,
    ClassicalSolvingConfig,
    QuantumSolvingConfig,
)

from qubosolver.solver.solver import Solver


__all__ = [
    "Solver",
    "SolverConfig",
    "DecompositionConfig",
    "DriveShapingConfig",
    "EmbeddingConfig",
    "ClassicalSolvingConfig",
    "QuantumSolvingConfig",
]
