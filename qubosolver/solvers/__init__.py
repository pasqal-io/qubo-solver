"""Solvers for QUBO problems.

This module provides quantum and classical solvers. Some of them can also be
used as post-processing steps to refine an existing solution.
"""

from __future__ import annotations

from qubosolver.solvers.enums import ClassicalAlgorithm
from qubosolver.solvers.classical.bitflip import iterative_bitflip_local_search
from qubosolver.solvers.quantum import analog_quantum_sampling
from qubosolver.solvers.classical.trivial import trivial_solution_search
from qubosolver.solvers.classical.cplex import cplex
from qubosolver.solvers.classical.tabu_search import tabu_search
from qubosolver.solvers.classical.simulated_annealing import simulated_annealing
from qubosolver.solvers.classical.random import random_sampling
from qubosolver.solvers.classical.brute_force import brute_force
from qubosolver.solvers.solver import Solver, QuboSolver
from qubosolver.solvers.config import Config, ClassicalConfig, DecompositionConfig, QuantumConfig

__all__ = [
    "ClassicalAlgorithm",
    "iterative_bitflip_local_search",
    "analog_quantum_sampling",
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
