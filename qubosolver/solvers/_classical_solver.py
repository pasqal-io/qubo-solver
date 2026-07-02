"""Classical QUBO solver implementations.

This module provides a family of classical optimisation solvers for QUBO
problems, all sharing the :class:`BaseClassicalSolver` interface.  The
correct solver is selected at runtime by :func:`get_classical_solver` based
on :attr:`~qubosolver.config.ClassicalConfig.classical_solver_type`.

Available solvers:

* :class:`CplexSolver` — exact MIP solver via IBM CPLEX (optional dependency).
* :class:`SimulatedAnnealingSolver` — stochastic temperature-cooling search.
* :class:`TabuSearchSolver` — neighbourhood search with a tabu memory.
* :class:`HybridSATabuSolver` — SA warm-start for Tabu Search; runs SA first
  and seeds Tabu Search with the best SA bitstring.
* :class:`RandomSolver` — uniform random sampling baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from qubosolver.config import ClassicalConfig
from qubosolver.types import QUBOInstance, QUBOSolution, ClassicalSolverType, torch_rng
from qubosolver import solvers


class BaseClassicalSolver(ABC):
    """Abstract base class for all classical QUBO solvers.

    Each concrete subclass implements a single optimisation strategy.
    Use :func:`get_classical_solver` to obtain the right subclass from a
    :class:`~qubosolver.config.ClassicalConfig` rather than instantiating
    subclasses directly.
    """

    def __init__(self, instance: QUBOInstance, config: ClassicalConfig):
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
    def solve(self) -> QUBOSolution:
        """Solve the QUBO problem and return a solution.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` containing the
            discovered bitstrings and their associated costs.
        """
        pass


class CplexSolver(BaseClassicalSolver):
    """QUBO solver backed by IBM CPLEX.

    Formulates the QUBO as a Mixed-Integer Program and delegates to the
    CPLEX solver.  Requires the optional ``cplex`` package to be installed;
    the import is deferred to :meth:`solve` so the rest of the module remains
    usable without it.

    Relevant :class:`~qubosolver.config.ClassicalConfig` fields:
    ``cplex_maxtime``, ``cplex_log_path``.
    """

    def solve(self) -> QUBOSolution:
        """Solve via CPLEX.

        Lazily imports :mod:`qubosolver.solvers.cplex` to avoid a hard
        dependency on the ``cplex`` package at module import time.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` with the optimal (or
            best feasible) bitstring found within ``config.cplex_maxtime``
            seconds.
        """
        from qubosolver.solvers import cplex

        log_path: str = self.config.cplex_log_path
        maxtime: float = self.config.cplex_maxtime

        return cplex(self.instance, maxtime=maxtime, log_path=log_path)


class SimulatedAnnealingSolver(BaseClassicalSolver):
    """QUBO solver using Simulated Annealing (SA).

    Explores the solution space by accepting uphill moves with a probability
    that decreases as temperature cools from ``sa_initial_temp`` to
    ``sa_final_temp``.

    Relevant :class:`~qubosolver.config.ClassicalConfig` fields:
    ``sa_seed``, ``sa_start``, ``sa_initial_temp``, ``sa_final_temp``,
    ``sa_cooling_rate``, ``sa_energy_tol``, ``sa_time_limit``,
    ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> QUBOSolution:
        """Solve via Simulated Annealing.

        When ``config.sa_start`` is ``None``, a single uniformly random
        bitstring is sampled (using ``config.sa_seed`` for reproducibility)
        and used as the starting point.  Otherwise ``config.sa_start`` is
        used directly.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
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
    """QUBO solver using Tabu Search.

    Performs neighbourhood search (single bit-flips) while maintaining a
    tabu list that forbids recently visited moves for ``tabu_tenure``
    iterations, preventing short cycles.

    Relevant :class:`~qubosolver.config.ClassicalConfig` fields:
    ``tabu_x0``, ``tabu_tenure``, ``tabu_max_no_improve``,
    ``tabu_time_limit``, ``max_iter``, ``max_bitstrings``.
    """

    def solve(self) -> QUBOSolution:
        """Solve via Tabu Search.

        When ``config.tabu_x0`` is ``None``, a uniformly random bitstring is
        sampled from the current global PyTorch RNG state and used as the
        starting point.  Otherwise ``config.tabu_x0`` is used directly.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` containing up to
            ``config.max_bitstrings`` best bitstrings found during the search.
        """
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
    """QUBO solver that combines Simulated Annealing with Tabu Search.

    Runs SA first to explore the solution space broadly, then seeds Tabu
    Search with the best bitstring found by SA for focused local refinement.
    This two-phase approach balances global exploration (SA) with precise
    local exploitation (Tabu).

    All ``sa_*`` and ``tabu_*`` fields of
    :class:`~qubosolver.config.ClassicalConfig` apply to their respective
    phases; ``max_iter`` and ``max_bitstrings`` are shared.
    """

    def solve(self) -> QUBOSolution:
        """Solve via SA warm-started Tabu Search.

        Internally creates a :class:`SimulatedAnnealingSolver` and a
        :class:`TabuSearchSolver` from copies of ``self.config`` with
        ``classical_solver_type`` overridden appropriately.  The best
        (lowest-cost) bitstring from SA is injected as ``tabu_x0`` for
        the Tabu phase.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` containing up to
            ``config.max_bitstrings`` best bitstrings found by Tabu Search,
            sorted by ascending cost.
        """
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
    """QUBO solver that returns uniformly random bitstrings.

    Useful as a baseline or for generating diverse starting points.
    Relevant :class:`~qubosolver.config.ClassicalConfig` field:
    ``max_bitstrings``.
    """

    def solve(self) -> QUBOSolution:
        """Sample random bitstrings from the current global PyTorch RNG state.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` with
            ``config.max_bitstrings`` uniformly sampled binary vectors and
            their corresponding QUBO costs.
        """
        rng = torch_rng().set_state(torch.get_rng_state())
        return solvers.random_solutions(
            self.instance, rng=rng, max_bitstrings=self.config.max_bitstrings
        )


def get_classical_solver(instance: QUBOInstance, config: ClassicalConfig) -> BaseClassicalSolver:
    """Return the appropriate classical solver for the given configuration.

    Dispatches on ``config.classical_solver_type`` (case-insensitive) to one
    of the five concrete solver classes:

    * ``"cplex"`` → :class:`CplexSolver`
    * ``"simulated_annealing"`` → :class:`SimulatedAnnealingSolver`
    * ``"simulated_annealing_tabu_search"`` → :class:`HybridSATabuSolver`
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
            known :class:`~qubosolver.types.ClassicalSolverType` value.
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
