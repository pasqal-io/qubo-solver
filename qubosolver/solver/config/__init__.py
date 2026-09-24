"""Configuration dataclasses for the solver and its sub-stages."""

from __future__ import annotations

from qubosolver.solver.config.config import (
    SolverConfig,
)
from qubosolver.solver.config.drive_shaping import Config as DriveShapingConfig
from qubosolver.solver.config.embedding import Config as EmbeddingConfig
from qubosolver.solver.config.solving import ClassicalConfig as ClassicalSolvingConfig
from qubosolver.solver.config.solving import QuantumConfig as QuantumSolvingConfig

__all__ = [
    "ClassicalSolvingConfig",
    "DriveShapingConfig",
    "EmbeddingConfig",
    "QuantumSolvingConfig",
    "SolverConfig",
]
