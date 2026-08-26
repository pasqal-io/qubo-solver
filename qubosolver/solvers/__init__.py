"""Solvers for QUBO problems.

This module provides quantum and classical solvers. Some of them can also be
used as post-processing steps to refine an existing solution.
"""

from __future__ import annotations

from qubosolver.solvers.enums import ClassicalAlgorithm
from qubosolver.solvers.classical import (
    brute_force,
    cplex,
    iterative_bitflip_local_search,
    random_sampling,
    simulated_annealing,
    tabu_search,
    trivial_solution_search,
)
from qubosolver.solvers.hybrid import drive_bayesian_search
from qubosolver.solvers.quantum import analog_quantum_sampling
from qubosolver.solvers import classical, hybrid, quantum
from qubosolver.solvers.solver import Solver, QuboSolver
from qubosolver.solvers.config import Config, ClassicalConfig, DecompositionConfig, QuantumConfig

__all__ = [
    "classical",
    "hybrid",
    "quantum",
    "ClassicalAlgorithm",
    "iterative_bitflip_local_search",
    "analog_quantum_sampling",
    "drive_bayesian_search",
    "trivial_solution_search",
    "cplex",
    "tabu_search",
    "simulated_annealing",
    "random_sampling",
    "brute_force",
    "Solver",
    "QuboSolver",
    "Config",
    "QuantumConfig",
    "ClassicalConfig",
    "DecompositionConfig",
]
