"""Classical QUBO solver algorithms.

Provides CPLEX, tabu search, simulated annealing, random sampling, and
brute-force solvers.
"""

from __future__ import annotations

from qubosolver.solvers.classical.config import Config
from qubosolver.solvers.classical import (
    brute_force,
    cplex,
    iterative_bitflip_local_search,
    random_sampling,
    simulated_annealing,
    tabu_search,
    trivial_solution_search,
)

__all__ = [
    "Config",
    "brute_force",
    "cplex",
    "iterative_bitflip_local_search",
    "random_sampling",
    "simulated_annealing",
    "tabu_search",
    "trivial_solution_search",
]
