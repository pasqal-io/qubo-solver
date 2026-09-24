"""Solving algorithms for QUBO problems.

This module provides quantum, hybrid and classical solvers. Some of them can also be
used as post-processing steps to refine an existing solution.
"""

from __future__ import annotations

from qubosolver.solving import classical, hybrid, quantum
from qubosolver.solving.classical import (
    brute_force,
    cplex,
    iterative_bitflip_local_search,
    random_sampling,
    simulated_annealing,
    tabu_search,
    trivial_solution_search,
)
from qubosolver.solving.hybrid import drive_bayesian_search
from qubosolver.solving.quantum import analog_quantum_sampling

__all__ = [
    "analog_quantum_sampling",
    "brute_force",
    "classical",
    "cplex",
    "drive_bayesian_search",
    "hybrid",
    "iterative_bitflip_local_search",
    "quantum",
    "random_sampling",
    "simulated_annealing",
    "tabu_search",
    "trivial_solution_search",
]
