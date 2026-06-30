"""
Module: classical_solver/classical_solver.py

Description:
    Implementation of multiple classical QUBO solvers.
    This module includes:
      - A solver based on CPLEX.
      - A solver using Simulated Annealing.
      - A solver using Tabu Search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from qubosolver.config import ClassicalConfig
from qubosolver.types import QUBOInstance, QUBOSolution, ClassicalSolverType, torch_rng
from qubosolver import solvers


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
            config (ClassicalConfig): Solver configuration
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
        from qubosolver.solvers import cplex

        log_path: str = self.config.cplex_log_path
        maxtime: float = self.config.cplex_maxtime

        return cplex(self.instance, maxtime=maxtime, log_path=log_path)


class SimulatedAnnealingSolver(BaseClassicalSolver):
    """
    QUBO solver using Simulated annealing solver.
    """

    def solve(self) -> QUBOSolution:
        rng = torch_rng(self.config.sa_seed)
        if self.config.sa_start is None:
            random_solution = solvers.random_solutions(self.instance, rng=rng, max_bitstrings=1)
            start = random_solution.bitstrings[0]
        else:
            start = self.config.sa_start

        simulated_annealing_solution = solvers.simulated_annealing(
            qubo=self.instance,
            top_k=self.config.max_bitstrings,
            max_iter=self.config.max_iter,
            initial_temp=self.config.sa_initial_temp,
            final_temp=self.config.sa_final_temp,
            cooling_rate=self.config.sa_cooling_rate,
            rng=rng,
            start=start,
            energy_tol=self.config.sa_energy_tol,
            time_limit=self.config.sa_time_limit,
        )
        return simulated_annealing_solution


class TabuSearchSolver(BaseClassicalSolver):
    """
    QUBO solver using Tabu search solver.
    """

    def solve(self) -> QUBOSolution:
        if self.config.tabu_x0 is None:
            assert self.instance.size
            rng = torch_rng().set_state(torch.get_rng_state())
            random_solution = solvers.random_solutions(self.instance, rng=rng, max_bitstrings=1)
            x0 = random_solution.bitstrings[0]
        else:
            x0 = self.config.tabu_x0
        tabu_search_solution = solvers.tabu_search(
            qubo=self.instance,
            start=x0,
            max_iter=self.config.max_iter,
            tabu_tenure=self.config.tabu_tenure,
            max_no_improve=self.config.tabu_max_no_improve,
            max_bitstrings=self.config.max_bitstrings,
            time_limit=self.config.tabu_time_limit,
        )
        return tabu_search_solution


class HybridSATabuSolver(BaseClassicalSolver):
    """
    QUBO solver using simulated annealing first followed by tabu search solver.

    Note: the starting point of tabu search is the best candidate
        obtained with simulated annealing.
    """

    def solve(self) -> QUBOSolution:
        config_sa = self.config.model_copy(
            update={"classical_solver_type": ClassicalSolverType.SIMULATED_ANNEALING}
        )
        sa = SimulatedAnnealingSolver(self.instance, config_sa)
        sa_solution = sa.solve()
        sa_solution.sort_by_cost()
        config_tabu = self.config.model_copy(
            update={
                "classical_solver_type": ClassicalSolverType.TABU_SEARCH,
                "tabu_x0": sa_solution.bitstrings[0],
            }
        )
        tabu = TabuSearchSolver(self.instance, config_tabu)
        tabu_sol = tabu.solve()
        tabu_sol.sort_by_cost()
        return tabu_sol


class RandomSolver(BaseClassicalSolver):
    """
    QUBO solver with random generation.
    """

    def solve(self) -> QUBOSolution:
        rng = torch_rng().set_state(torch.get_rng_state())
        return solvers.random_solutions(
            self.instance, rng=rng, max_bitstrings=self.config.max_bitstrings
        )


def get_classical_solver(instance: QUBOInstance, config: ClassicalConfig) -> BaseClassicalSolver:
    """
    Returns the appropriate QUBO solver based on the configuration.

    Args:
        instance (QUBOInstance): The QUBO problem instance.
        config (ClassicalConfig): Classical solver configuration containing
            the ``classical_solver_type`` field.

    Returns:
        BaseClassicalSolver: An instance of the selected QUBO solver.

    Raises:
        ValueError: If the requested solver type is not supported.
    """
    solver_type = config.classical_solver_type
    solver_type = solver_type.lower()

    if solver_type == ClassicalSolverType.CPLEX:
        return CplexSolver(instance, config)
    if solver_type == ClassicalSolverType.SIMULATED_ANNEALING:
        return SimulatedAnnealingSolver(instance, config)
    if solver_type == ClassicalSolverType.SIMULATED_ANNEALING_TABU_SEARCH:
        return HybridSATabuSolver(instance, config)
    if solver_type == ClassicalSolverType.TABU_SEARCH:
        return TabuSearchSolver(instance, config)
    if solver_type == ClassicalSolverType.RANDOM:
        return RandomSolver(instance, config)

    raise ValueError(f"Solver type not supported: {solver_type}")
