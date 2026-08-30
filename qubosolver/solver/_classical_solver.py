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

from qubosolver.types import Instance, Solution, torch_rng
from qubosolver import solvers


class BaseClassicalSolver(ABC):
    """Abstract base class for all classical QUBO solvers.

    Each concrete subclass implements a single optimisation strategy.
    Use `get_classical_solver` to obtain the right subclass from a
    :class:`~qubosolver.solvers.config.classical.Config` rather than instantiating
    subclasses directly.
    """

    def __init__(self, instance: Instance, config: solvers.classical.Config):
        """Initialise the solver with a QUBO instance and configuration.

        Args:
            instance: The QUBO problem instance to solve.
            config: Classical solver configuration.  The relevant fields
                depend on the concrete subclass (e.g. ``cplex_maxtime`` for
                :class:`CplexSolver`, ``sa_*`` fields for
                :class:`SimulatedAnnealingSolver`).
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

    Relevant :class:`~qubosolver.solvers.config.classical.Config` fields:
    ``cplex_maxtime``, ``cplex_log_path``.
    """

    def solve(self) -> Solution:
        """Solve via CPLEX.

        Lazily imports :mod:`qubosolver.solvers.cplex` to avoid a hard
        dependency on the ``cplex`` package at module import time.

        Returns:
            A :class:`~qubosolver.types.Solution` with the optimal (or
            best feasible) bitstring found within ``config.cplex_maxtime``
            seconds.
        """
        from qubosolver.solvers import cplex

        log_path: str = self.config.cplex_log_path
        maxtime: float = self.config.cplex_maxtime

        return cplex.solve(self.instance, maxtime=maxtime, log_path=log_path)


class SimulatedAnnealingSolver(BaseClassicalSolver):
    """QUBO solver using Simulated Annealing (SA).

    Explores the solution space by accepting uphill moves with a probability
    that decreases as temperature cools from ``sa_initial_temp`` to
    ``sa_final_temp``.

    Relevant :class:`~qubosolver.solvers.config.classical.Config` fields:
    ``sa_seed``, ``sa_start``, ``sa_initial_temp``, ``sa_final_temp``,
    ``sa_cooling_rate``, ``sa_time_limit``,
    ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> Solution:
        """Solve via Simulated Annealing.

        When ``config.sa_start`` is ``None``, a single uniformly random
        bitstring is sampled (using ``config.sa_seed`` for reproducibility)
        and used as the starting point.  Otherwise ``config.sa_start`` is
        used directly.

        Returns:
            A :class:`~qubosolver.types.Solution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
        rng = torch_rng(self.config.sa_seed)
        if self.config.sa_start is None:
            random_solution = solvers.random_sampling.solve(self.instance, rng=rng, max_bitstrings=1)
            start = random_solution.bitstrings[0]
        else:
            start = self.config.sa_start

        return solvers.simulated_annealing.solve(
            instance=self.instance,
            top_k=self.config.max_bitstrings,
            max_iter=self.config.max_iter,
            initial_temp=self.config.sa_initial_temp,
            final_temp=self.config.sa_final_temp,
            cooling_rate=self.config.sa_cooling_rate,
            rng=rng,
            start=start.unsqueeze(0),
            time_limit=self.config.sa_time_limit,
            stats="per_run",
        )


class TabuSearchSolver(BaseClassicalSolver):
    """QUBO solver using Tabu Search.

    Performs neighbourhood search (single bit-flips) while maintaining a
    tabu list that forbids recently visited moves for ``tabu_tenure``
    iterations, preventing short cycles.

    Relevant :class:`~qubosolver.solvers.config.classical.Config` fields:
    ``tabu_x0``, ``tabu_tenure``, ``tabu_max_no_improve``,
    ``tabu_time_limit``, ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> Solution:
        """Solve via Tabu Search.

        When ``config.tabu_x0`` is ``None``, a uniformly random bitstring is
        sampled from the current global PyTorch RNG state and used as the
        starting point.  Otherwise ``config.tabu_x0`` is used directly.

        Returns:
            A :class:`~qubosolver.types.Solution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
        if self.config.tabu_x0 is None:
            assert self.instance.size
            rng = torch_rng().set_state(torch.get_rng_state())
            random_solution = solvers.random_sampling.solve(
                self.instance, rng=rng, max_bitstrings=self.config.max_bitstrings
            )
            x0 = random_solution.bitstrings
        else:
            x0 = self.config.tabu_x0
        tabu_search_solution = solvers.tabu_search.solve(
            qubo=self.instance,
            start=x0,
            max_iter=self.config.max_iter,
            tabu_tenure=self.config.tabu_tenure,
            max_no_improve=self.config.tabu_max_no_improve,
            time_limit=self.config.tabu_time_limit,
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
        return solvers.classical.random_sampling.solve(
            self.instance, rng=rng, max_bitstrings=self.config.max_bitstrings
        )


def get_classical_solver(instance: Instance, config: solvers.classical.Config) -> BaseClassicalSolver:
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
            known :class:`~qubosolver.solvers.classical.Algorithm` value.
    """
    solver_type = config.algorithm
    solver_type = solver_type.lower()

    if solver_type == solvers.classical.Algorithm.CPLEX:
        return CplexSolver(instance, config)
    if solver_type == solvers.classical.Algorithm.SIMULATED_ANNEALING:
        return SimulatedAnnealingSolver(instance, config)
    if solver_type == solvers.classical.Algorithm.TABU_SEARCH:
        return TabuSearchSolver(instance, config)
    if solver_type == solvers.classical.Algorithm.RANDOM_SAMPLING:
        return RandomSolver(instance, config)

    raise ValueError(f"Solver type not supported: {solver_type}")
