"""Enum types used by classical solver selection."""

from __future__ import annotations

from qubosolver.types._enums import _StrEnum


class ClassicalAlgorithm(_StrEnum):
    """Type of classical solver used as a backend for QUBO optimization."""

    TABU_SEARCH = "tabu_search"
    """Tabu search metaheuristic that avoids recently visited solutions."""
    SIMULATED_ANNEALING = "simulated_annealing"
    """Simulated annealing algorithm that probabilistically accepts worse solutions to escape local minima."""
    CPLEX = "cplex"
    """IBM CPLEX exact solver; requires a valid CPLEX installation and licence."""
    RANDOM = "random"
    """Randomly samples solutions; useful as a baseline or for testing."""
