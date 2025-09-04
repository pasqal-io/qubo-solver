"""
Module: classical_solver/classical_solver.py

Description:
    Implementation of multiple classical QUBO solvers.
    This module includes:
      - A solver based on CPLEX.
      - A solver using D-Wave Simulated Annealing.
      - A solver using D-Wave Tabu Search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cplex
import torch

# D-Wave imports
from dwave.samplers import SimulatedAnnealingSampler

# For Tabu Search, we need:
from dwave.samplers.tabu import TabuSampler

# QUBO solver imports
from qubosolver import QUBOInstance, QUBOSolution

# Import conversion utilities from classical_solver_conversion_tools.
from qubosolver.classical_solver.classical_solver_conversion_tools import (
    qubo_instance_to_sparsepairs,
)
from qubosolver.classical_solver.classical_solver_conversion_tools import (
    run_sampler as conversion_run_sampler,
)


# =============================================================================
# Abstract base class and solver implementations
# =============================================================================
class BaseClassicalSolver(ABC):
    """
    Abstract base class for all classical QUBO solvers.
    Stores the QUBO instance and an optional configuration dictionary.
    """

    def __init__(self, instance: QUBOInstance, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the solver with a given QUBO instance and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem instance to solve.
            config (Optional[Dict[str, Any]]): Solver configuration
            (e.g., cplex_maxtime, cplex_log_path, classical_solver_type).
        """
        self.instance = instance
        self.config = config if config is not None else {}

    @abstractmethod
    def solve(self) -> QUBOSolution:
        """
        Abstract method to solve the QUBO problem.

        Returns:
            QUBOSolution: The solution object containing bitstrings,
            costs, and optionally counts and probabilities.
        """
        pass


# -----------------------------------------------------------------------------
# CPLEX-based QUBO solver implementation.
# -----------------------------------------------------------------------------
class CplexSolver(BaseClassicalSolver):
    """
    QUBO solver based on CPLEX.
    """

    def solve(self) -> QUBOSolution:
        # Extract configuration parameters using new keys.
        log_path: str = self.config.get("cplex_log_path", "solver.log")
        maxtime: float = self.config.get("cplex_maxtime", 600.0)

        if self.instance.coefficients is None:
            raise ValueError("The QUBO instance does not contain coefficients.")

        # Determine the number of variables.
        N: int = self.instance.coefficients.shape[0]
        # If there are no variables, return an empty solution.
        if N == 0:
            bitstring_tensor = torch.empty((0, 0), dtype=torch.float32)
            cost_tensor = torch.empty((0,), dtype=torch.float32)
            return QUBOSolution(bitstrings=bitstring_tensor, costs=cost_tensor)

        # Convert the coefficient matrix into CPLEX sparse pairs format using the conversion tool.
        sparsepairs: List[cplex.SparsePair] = qubo_instance_to_sparsepairs(self.instance)

        # Open a log file.
        log_file = open(log_path, "w")
        problem = cplex.Cplex()

        # Redirect logging streams.
        problem.set_log_stream(log_file)
        problem.set_error_stream(log_file)
        problem.set_warning_stream(log_file)
        problem.set_results_stream(log_file)

        problem.parameters.timelimit.set(maxtime)
        problem.objective.set_sense(problem.objective.sense.minimize)

        # Add binary variables.
        problem.variables.add(types="B" * N)

        # Set the quadratic objective.
        problem.objective.set_quadratic(sparsepairs)

        problem.solve()

        # Retrieve solution.
        solution_values = problem.solution.get_values()
        solution_cost = problem.solution.get_objective_value()

        log_file.close()

        # Convert the solution into a QUBOSolution.
        bitstring_tensor = torch.tensor([[int(b) for b in solution_values]], dtype=torch.float32)
        cost_tensor = torch.tensor([solution_cost], dtype=torch.float32)

        return QUBOSolution(bitstrings=bitstring_tensor, costs=cost_tensor)


# -----------------------------------------------------------------------------
# D-Wave Simulated Annealing (SA) solver implementation.
# -----------------------------------------------------------------------------
class DwaveSASolver(BaseClassicalSolver):
    """
    QUBO solver using D-Wave's Simulated Annealing sampler.
    """

    def solve(self) -> QUBOSolution:
        # Initialize the D-Wave Simulated Annealing sampler.
        sampler = SimulatedAnnealingSampler()
        # Use the conversion tool's run_sampler (which returns a QUBOSolution).
        solution: QUBOSolution = conversion_run_sampler(sampler, self.instance)
        return solution


# -----------------------------------------------------------------------------
# D-Wave Tabu Search solver implementation.
# -----------------------------------------------------------------------------
class DwaveTabuSolver(BaseClassicalSolver):
    """
    QUBO solver using D-Wave's Tabu Search heuristic.
    """

    def solve(self) -> QUBOSolution:
        # Initialize the D-Wave Tabu Search sampler.
        sampler = TabuSampler()
        solution: QUBOSolution = conversion_run_sampler(sampler, self.instance)
        return solution


# =============================================================================
# Factory function to select the appropriate solver based on configuration.
# =============================================================================
def get_classical_solver(
    instance: QUBOInstance, config: Optional[Dict[str, Any]] = None
) -> BaseClassicalSolver:
    """
    Returns the appropriate QUBO solver based on the configuration.

    Args:
        instance (QUBOInstance): The QUBO problem instance.
        config (Optional[Dict[str, Any]]): Configuration,
          possibly including 'classical_solver_type'.

    Returns:
        BaseClassicalSolver: An instance of a QUBO solver.

    Raises:
        ValueError: If the requested solver type is not supported.
    """
    solver_type = config.get("classical_solver_type", "cplex") if config is not None else "cplex"
    solver_type = solver_type.lower()

    if solver_type == "cplex":
        return CplexSolver(instance, config)
    elif solver_type == "dwave_sa":
        return DwaveSASolver(instance, config)
    elif solver_type == "dwave_tabu":
        return DwaveTabuSolver(instance, config)
    else:
        raise ValueError(f"Solver type not supported: {solver_type}")
