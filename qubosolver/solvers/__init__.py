"""Solvers for QUBO problems.

This package provides quantum and classical solvers. Some of them can also be
used as post-processing steps to refine an existing solution.
"""

from __future__ import annotations

from qubosolver.solvers.bitflip import iterative_bitflip_local_search
from qubosolver.solvers.quantum import analog_quantum_sample
from qubosolver.solvers.trivial import trivial_solution_search
from qubosolver.solvers.cplex import cplex
from qubosolver.solvers.tabu_search import tabu_search
from qubosolver.solvers.simulated_annealing import simulated_annealing
from qubosolver.solvers.random import random_solutions
from qubosolver.solvers.solver import Solver, QuboSolver

__all__ = [
    "iterative_bitflip_local_search",
    "analog_quantum_sample",
    "trivial_solution_search",
    "cplex",
    "tabu_search",
    "simulated_annealing",
    "random_solutions",
    "Solver",
    "QuboSolver",
]
