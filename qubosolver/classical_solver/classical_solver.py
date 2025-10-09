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
from typing import List

import cplex
import torch

from qubosolver.config import ClassicalConfig
from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.classical_solver.simulated_annealing import qubo_simulated_annealing
from qubosolver.classical_solver.tabu_search import qubo_tabu_search
from qubosolver.classical_solver.classical_solver_conversion_tools import (
    qubo_instance_to_sparsepairs,
)


class BaseClassicalSolver(ABC):
    """
    Abstract base class for all classical QUBO solvers.
    Stores the QUBO instance and an optional configuration dictionary.
    """

    def __init__(self, instance: QUBOInstance, config: ClassicalConfig):
        """
        Initializes the solver with a given QUBO instance and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem instance to solve.
            config (Optional[Dict[str, Any]]): Solver configuration
            (e.g., cplex_maxtime, cplex_log_path, classical_solver_type).
        """
        self.instance = instance
        self.config = config

    @abstractmethod
    def solve(self) -> QUBOSolution:
        """
        Abstract method to solve the QUBO problem.

        Returns:
            QUBOSolution: The solution object containing bitstrings,
            costs, and optionally counts and probabilities.
        """
        pass


class CplexSolver(BaseClassicalSolver):
    """
    QUBO solver based on CPLEX.
    """

    def solve(self) -> QUBOSolution:
        # Extract configuration parameters using new keys.
        log_path: str = self.config.cplex_log_path
        maxtime: float = self.config.cplex_maxtime

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


class SimulatedAnnealingSolver(BaseClassicalSolver):
    """
    QUBO solver using Simulated annealing solver.
    """

    def solve(self) -> QUBOSolution:
        simulated_annealing_solution = qubo_simulated_annealing(
            qubo=self.instance,
            max_iter=self.config.max_iter,
            initial_temp=self.config.sa_initial_temp,
            final_temp=self.config.sa_final_temp,
            alpha=self.config.sa_alpha,
        )
        return simulated_annealing_solution


class TabuSearchSolver(BaseClassicalSolver):
    """
    QUBO solver using Tabu search solver.
    """

    def solve(self) -> QUBOSolution:
        if not self.config.tabu_x0:
            assert self.instance.size
            x0 = torch.randint(0, 2, size=(self.instance.size,))
        else:
            x0 = self.config.tabu_x0

        tabu_search_solution = qubo_tabu_search(
            qubo=self.instance,
            x0=x0,
            max_iter=self.config.max_iter,
            tabu_tenure=self.config.tabu_tenure,
            max_no_improve=self.config.tabu_max_no_improve,
        )
        return tabu_search_solution


def get_classical_solver(instance: QUBOInstance, config: ClassicalConfig) -> BaseClassicalSolver:
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
    solver_type = config.classical_solver_type
    solver_type = solver_type.lower()

    if solver_type == "cplex":
        return CplexSolver(instance, config)
    if solver_type == "simulated_annealing":
        return SimulatedAnnealingSolver(instance, config)
    if solver_type == "tabu_search":
        return TabuSearchSolver(instance, config)

    raise ValueError(f"Solver type not supported: {solver_type}")
