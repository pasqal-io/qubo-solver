"""Quantum solvers based on analog sampling on Pasqal devices."""

from __future__ import annotations

from qubosolver.solvers.quantum.config import Config
from qubosolver.solvers.quantum import (
    analog_quantum_sampling,
)

__all__ = [
    "Config",
    "analog_quantum_sampling",
]
