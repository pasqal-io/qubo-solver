"""Classical QUBO solver implementations.

This module provides a family of classical optimisation solvers for QUBO
problems, all sharing the :class:`BaseClassicalSolver` interface.  The
correct solver is selected at runtime by `get_classical_solver` based
on :attr:`~qubosolver.solvers.config.classical.Config.classical_solver_type`.

Available solvers:

* :class:`CplexSolver` — exact MIP solver via IBM CPLEX (optional dependency).
* :class:`SimulatedAnnealingSolver` — stochastic temperature-cooling search.
* :class:`TabuSearchSolver` — neighbourhood search with a tabu memory.
* :class:`RandomSolver` — uniform random sampling baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from qubosolver import solving
from qubosolver.types import Instance, Solution, bitstrings, torch_rng

from .config import ClassicalSolvingConfig


class BaseClassicalSolver(ABC):
    """Abstract base class for all classical QUBO solvers.

    Each concrete subclass implements a single optimisation strategy.
    Use `get_classical_solver` to obtain the right subclass from a
    :class:`~qubosolver.solvers.config.classical.Config` rather than instantiating
    subclasses directly.
    """

    def __init__(self, instance: Instance, config: ClassicalSolvingConfig) -> None:
        """Initialise the solver with a QUBO instance and configuration.

        Args:
            instance: The QUBO problem instance to solve.
            config: Classical solver configuration.  The relevant fields
                depend on the concrete subclass (e.g. ``time_limit`` for
                :class:`CplexSolver`, :class:`SimulatedAnnealingSolver`,
                and :class:`TabuSearchSolver`).
        """
        self.instance = instance
        self.config = config

    @abstractmethod
    def solve(self) -> Solution:
        """Solve the QUBO problem and return a solution.

        Returns:
            A :class:`~qubosolver.types.Solution` containing the
            discovered bitstrings and their associated costs.
        """
        pass


class CplexSolver(BaseClassicalSolver):
    """QUBO solver backed by IBM CPLEX.

    Formulates the QUBO as a Mixed-Integer Program and delegates to the
    CPLEX solver.  Requires the optional ``cplex`` package to be installed;
    the import is deferred to :meth:`solve` so the rest of the module remains
    usable without it.

    Relevant :class:`~qubosolver.solvers.config.classical.Config` field:
    ``time_limit``.
    """

    def solve(self) -> Solution:
        """Solve via CPLEX.

        Lazily imports :mod:`qubosolver.solvers.cplex` to avoid a hard
        dependency on the ``cplex`` package at module import time.

        Returns:
            A :class:`~qubosolver.types.Solution` with the optimal (or
            best feasible) bitstring found within ``config.time_limit``
            seconds.
        """
        from qubosolver.solving import cplex

        return cplex.solve(self.instance, time_limit=self.config.time_limit)


class SimulatedAnnealingSolver(BaseClassicalSolver):
    """QUBO solver using Simulated Annealing (SA).

    Explores the solution space by accepting uphill moves with a probability
    that decreases as the temperature cools over the course of the search.

    Relevant :class:`~qubosolver.solvers.config.classical.Config` fields:
    ``time_limit``, ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> Solution:
        """Solve via Simulated Annealing.

        A single uniformly random bitstring is sampled as the starting point.

        Returns:
            A :class:`~qubosolver.types.Solution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
        starts = bitstrings.rand(1, self.instance.size)

        return solving.simulated_annealing.solve(
            instance=self.instance,
            top_k=self.config.max_bitstrings,
            max_iter=self.config.max_iter,
            starts=starts,
            time_limit=self.config.time_limit,
            stats="per_run",
        )


class TabuSearchSolver(BaseClassicalSolver):
    """QUBO solver using Tabu Search.

    Performs neighbourhood search (single bit-flips) while maintaining a
    tabu list that forbids recently visited moves for a number of
    iterations, preventing short cycles.

    Relevant :class:`~qubosolver.solvers.config.classical.Config` fields:
    ``time_limit``, ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> Solution:
        """Solve via Tabu Search.

        A single uniformly random bitstring is sampled as the starting point.

        Returns:
            A :class:`~qubosolver.types.Solution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
        starts = bitstrings.rand(1, self.instance.size)
        tabu_search_solution = solving.tabu_search.solve(
            instance=self.instance,
            starts=starts,
            max_iter=self.config.max_iter,
            time_limit=self.config.time_limit,
        )
        return tabu_search_solution


class RandomSolver(BaseClassicalSolver):
    """QUBO solver that returns uniformly random bitstrings.

    Useful as a baseline or for generating diverse starting points.
    Relevant :class:`~qubosolver.solvers.config.classical.Config` field:
    ``max_bitstrings``.
    """

    def solve(self) -> Solution:
        """Sample random bitstrings from the current global PyTorch RNG state.

        Returns:
            A :class:`~qubosolver.types.Solution` with
            ``config.max_bitstrings`` uniformly sampled binary vectors and
            their corresponding QUBO costs.
        """
        rng = torch_rng().set_state(torch.get_rng_state())
        return solving.classical.random_sampling.solve(
            self.instance, rng=rng, max_bitstrings=self.config.max_bitstrings
        )


def get_classical_solver(instance: Instance, config: ClassicalSolvingConfig) -> BaseClassicalSolver:
    """Return the appropriate classical solver for the given configuration.

    Dispatches on ``config.classical_solver_type`` (case-insensitive) to one
    of the four concrete solver classes:

    * ``"cplex"`` → :class:`CplexSolver`
    * ``"simulated_annealing"`` → :class:`SimulatedAnnealingSolver`
    * ``"tabu_search"`` → :class:`TabuSearchSolver`
    * ``"random"`` → :class:`RandomSolver`

    Args:
        instance: The QUBO problem instance to solve.
        config: Classical solver configuration.  ``classical_solver_type``
            determines which solver is returned; other fields are forwarded
            to the chosen solver.

    Returns:
        A concrete :class:`BaseClassicalSolver` ready to have
        :meth:`~BaseClassicalSolver.solve` called.

    Raises:
        ValueError: If ``config.classical_solver_type`` does not match any
            known :class:`~qubosolver.solver.config.solving.ClassicalAlgorithm` value.
    """
    solver_type = config.algorithm

    match solver_type:
        case "cplex":
            return CplexSolver(instance, config)
        case "simulated_annealing":
            return SimulatedAnnealingSolver(instance, config)
        case "tabu_search":
            return TabuSearchSolver(instance, config)
        case "random_sampling":
            return RandomSolver(instance, config)
        case _:
            raise ValueError(f"Invalid solver name: {solver_type}")
