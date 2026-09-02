"""Concrete QUBO solver implementations built on top of [`BaseSolver`][].

This module provides three internal solver classes and the public
[`Solver`][qubosolver.Solver]:

* [`Solver`][qubosolver.Solver] — public entry point.  Its concrete solving
  strategy is chosen from [`Config`][] at construction time, from
  the three solvers below.
* `_QuboSolverQuantum` — runs the full quantum pipeline: embedding →
  drive shaping → analog quantum sampling → postprocessing.
* `_QuboSolverClassical` — solves using a classical optimizer
  (CPLEX, Simulated Annealing, Tabu Search, …) via
  `~qubosolver.solver.get_classical_solver`.
* `_DecomposeQuboSolver` — recursively decomposes a large QUBO into
  device-sized subproblems, solves each with a sub-solver, and merges the
  partial solutions.
"""

from __future__ import annotations

from collections.abc import Callable

import logging
import torch
from typing_extensions import TypeAlias
from copy import deepcopy

import qoolqit

from qubosolver import Solution, Instance, transforms, torch_rng, drive_shaping
from .config import DecompositionConfig, SolverConfig, ClassicalSolvingConfig
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal
from ._basesolver import BaseSolver
from ._classical_solver import get_classical_solver
from ._embedder import _get_embedder
from ._drive_shaper import _get_drive_shaper

logger = logging.getLogger(__name__)


class Solver(BaseSolver):
    """A QUBO solver.

    Its concrete solving strategy (quantum, classical, or decomposition)
    is chosen from [`SolverConfig`][] at construction time.

    Example:
        ```python
        from qubosolver import Instance, Solver, SolverConfig

        instance = Instance(matrix)
        config = SolverConfig()
        solver = Solver(instance, config)
        solution = solver.solve()
        ```
    """

    def __init__(self, instance: Instance, config: SolverConfig = SolverConfig()):
        """Initialize the solver.

        Args:
            instance: The QUBO problem to solve.
            config: Solver configuration controlling which solving strategy
                is used and how it behaves.
        """
        super().__init__(instance, config)
        self._solver: BaseSolver

        if config.decompose:
            if self.config.solving_mode == "quantum":
                solver_factory: type[BaseSolver] = _QuboSolverQuantum
            else:
                solver_factory = _QuboSolverClassical
            self._solver = _DecomposeQuboSolver(instance, self.config, solver_factory)

        elif self.config.solving_mode == "quantum":
            self._solver = _QuboSolverQuantum(instance, config)
        else:
            self._solver = _QuboSolverClassical(instance, config)

    def _embedding(self) -> qoolqit.Register:
        """Embed QUBO variables onto physical atom positions.

        Returns:
            The `qoolqit.Register` for the current instance.
        """
        return self._solver._embedding()

    def _drive(self, embedding: qoolqit.Register) -> tuple[qoolqit.Drive, Solution]:
        """Generate the pulse drive schedule for the given embedding.

        Args:
            embedding: The register layout produced by [`_embedding`][].

        Returns:
            A ``(Drive, Solution)`` tuple.
        """
        return self._solver._drive(embedding)

    def solve(self) -> Solution:
        """Solve the QUBO instance.

        Returns:
            The solution.
        """
        return self._solver.solve()


class _QuboSolverQuantum(BaseSolver):
    """Quantum QUBO solver using embedding, drive shaping, and analog sampling.

    Orchestrates the full quantum pipeline:
    embedding → drive shaping → analog quantum execution → postprocessing.

    Constraints:

    * All off-diagonal QUBO coefficients must be non-negative (Rydberg
      blockade physics cannot encode attractive interactions).
    * Problem size is capped at 80 variables (device atom-number limit).
    """

    def __init__(self, instance: Instance, config: SolverConfig = SolverConfig()):
        """Initialise the quantum solver.

        Args:
            instance: The QUBO problem to solve.  Raises immediately if any
                off-diagonal coefficient is negative or the problem exceeds
                80 variables.
            config: Solver settings (backend, device, embedding, drive
                shaping, etc.).  Defaults to
                ``Config(use_quantum=True)`` when ``None``.

        Raises:
            ValueError: If ``instance.size > 80``.
        """

        super().__init__(instance, config or Config())

        if _has_negative_offdiagonal(instance.matrix) and not self.config.preprocessing:
            logger.warning(
                "Negative off-diagonal coefficients detected. Without preprocessing, this "
                "instance will be embedded via automatic zeroing (an approximation). Set "
                "config.preprocessing=True to try bit-flip preprocessing instead, which "
                "is likely to give better results."
            )

        self._check_size_limit()

        self.backend = self.config.quantum.backend
        self.embedder = _get_embedder(self.instance, self.config.quantum, self.backend)
        self.drive_shaper = _get_drive_shaper(self.instance, self.config.quantum, self.backend)

        self._cached_register: qoolqit.Register | None = None
        self._cached_drive: qoolqit.Drive | None = None

    def _check_size_limit(self) -> None:
        """Raise if the QUBO instance exceeds the 80-variable device limit.

        Raises:
            ValueError: If ``self.instance.size > 80``.
        """
        if self.instance.size > 80:
            raise ValueError(
                f"QUBO size {self.instance.size}×{self.instance.size}"
                + " exceeds the maximum supported size of 80×80."
            )

    def _embedding(self) -> qoolqit.Register:
        """Embed QUBO variables onto physical atom positions.

        Calls the configured embedder (BLaDE or Greedy) and caches the
        result in ``self._cached_register``.

        Returns:
            The atom `qoolqit.Register` layout for the current
                instance.
        """
        self.embedder.instance = self.instance
        self._cached_register = self.embedder.embed()
        return self._cached_register

    def _drive(self, embedding: qoolqit.Register) -> tuple[qoolqit.Drive, Solution]:
        """Generate the pulse drive schedule for the given embedding.

        Calls the configured drive shaper (proportional-diagonal or optimised) and
        caches the drive in ``self._cached_drive``.

        Args:
            embedding: The atom register layout produced by [`_embedding`][].

        Returns:
            A 2-tuple of:

                * `qoolqit.Drive` — the pulse schedule for quantum
                    execution.
                * [`Solution`][qubosolver.Solution] — an initial solution
                    produced as a by-product of drive shaping (may be empty for
                    the proportional-diagonal shaper).
        """
        self.drive_shaper.instance = self.instance
        drive, qubo_solution = self.drive_shaper.generate(embedding)

        self._cached_drive = drive
        return drive, qubo_solution

    def solve(self) -> Solution:
        """Execute the full quantum pipeline and return the best solution.

        Returns:
            The final [`Solution`][qubosolver.Solution], sorted by
                ascending cost with probabilities computed.
        """
        assert self.config.solving_mode == "quantum"

        # 1) try trivial else delegate to quantum solver
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial:
                return trivial
        self._check_size_limit()

        # 2) Apply preprocessing if requested
        self._preprocess()

        if _has_negative_offdiagonal(self.instance.matrix):
            logger.info(
                "Bit-flip preprocessing could not remove all negative off-diagonal "
                "coefficients; zeroing the remainder as an approximation before embedding."
            )
            self.instance = transforms.zeroing.apply(self.instance)
            self._update_instance(self.instance)

        embedding = self._embedding()
        drive, solution = self._drive(embedding)

        if not solution or self.config.quantum.drive_shaping.algorithm != "bayesian_search":
            solution = self._execute(drive, embedding)

        if isinstance(self.instance, transforms.zeroing.Instance):
            solution = transforms.zeroing.lift(solution, self.instance)
            self._update_instance(self.instance._parent_instance)

        # Post-process fixations of the preprocessing and restore the original QUBO
        solution = self._post_process_fixation(solution)
        solution = self._post_process(solution)

        solution._update(self.instance)

        return solution


class _QuboSolverClassical(BaseSolver):
    """Classical QUBO solver.

    Delegates solving to the classical optimiser selected by
    `~qubosolver.solvers.get_classical_solver` (CPLEX, Simulated
    Annealing, Tabu Search, or Random).

    Embedding and drive generation are not meaningful for classical solvers;
    both methods are no-ops that return ``None``.
    """

    def __init__(self, instance: Instance, config: SolverConfig = SolverConfig()):
        super().__init__(instance, config)

    def _embedding(self) -> qoolqit.Register:
        """No-op — classical solvers do not require an atom register.

        Returns:
            ``None``.
        """
        return  # type: ignore[return-value]

    def _drive(self, embedding: qoolqit.Register) -> tuple:
        """No-op — classical solvers do not use a pulse drive.

        Returns:
            ``None``.
        """
        return  # type: ignore[return-value]

    def solve(self) -> Solution:
        """Solve the QUBO using the configured classical solver.

        Returns:
            The final [`Solution`][qubosolver.Solution].
        """
        # 1) try trivial
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial:
                return trivial

        self._preprocess()

        solution = Solution()

        if self.instance.size != 0:
            classical_solver = get_classical_solver(self.instance, self.config.classical)
            solution = (
                classical_solver.solve()
            )  # This is a reduced solution if pre-procesing is applied

        solution = self._post_process_fixation(solution)
        solution = self._post_process(solution)

        return solution


class _DecomposeQuboSolver(BaseSolver):
    """Device-aware QUBO decomposition solver.

    Treats the QUBO as a graph (variables = vertices, coefficients =
    weighted edges), iteratively extracts device-sized subgraphs using
    geometric search, solves each subproblem with a configurable
    ``solver_factory``, and merges the partial solutions back into a
    global solution.  The final tail of variables (those that fall below
    `DecompositionConfig.decompose_stop_number`)
    is always solved classically.

    Constraints:

    * All off-diagonal QUBO coefficients must be non-negative (same
      physical constraint as `_QuboSolverQuantum`).
    * Exactly **one** bitstring is returned (design choice of the
      decomposition algorithm).
    """

    def __init__(
        self,
        instance: Instance,
        config: SolverConfig | None,
        solver_factory: Callable[[Instance, SolverConfig], BaseSolver],
    ):
        """Initialise the decomposition solver.

        Args:
            instance: The QUBO problem to decompose and solve.  All
                off-diagonal coefficients must be non-negative.
            config: Solver settings (backend, device, decomposition
                parameters, etc.).  Defaults to
                ``Config(use_quantum=True)`` when ``None``.
            solver_factory: A callable (typically
                `_QuboSolverQuantum` or `_QuboSolverClassical`) that
                constructs the sub-solver used to solve each extracted
                subproblem.

        Raises:
            ValueError: If any off-diagonal coefficient in
                ``instance.matrix`` is negative.
        """
        if (instance.matrix[~torch.eye(*instance.matrix.shape, dtype=torch.bool)] < 0).any():
            raise ValueError("Decomposition does not handle off-diagonal negative coefficients")

        # default is a quantum solver as we apply device-dependent decomposition
        super().__init__(
            Instance(instance.matrix),
            config or SolverConfig(),
        )
        self._solver_factory = solver_factory

        self.decomposition_config: DecompositionConfig = (
            self.config.decompose or DecompositionConfig()
        )

        # A cached version of `config` that we're going
        # to use for problems we do not wish to decompose.
        self._config_subproblems = deepcopy(self.config)
        self._config_subproblems.decompose = None

        self._decomposition = [list(range(instance.size))]

    def _embedding(self) -> qoolqit.Register:
        """Not supported — decomposition manages embeddings per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def _drive(self, embedding: qoolqit.Register) -> tuple:
        """Not supported — decomposition manages drives per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def solve(self) -> Solution:
        """Solve the QUBO by iterative decomposition into device-sized subproblems.

        Returns:
            A [`Solution`][qubosolver.Solution] containing exactly one
                bitstring — the merged result of all subproblem solutions.
        """
        # Create a local Generator that inherits whatever seeding you did via torch.manual_seed(...) (copies the current global RNG state).
        rng = torch_rng().set_state(torch.get_rng_state())
        self.number_iterations = 0
        assert self.instance.size  # nosec B101
        if self.instance.size <= self.decomposition_config.decompose_stop_number:
            solver = _QuboSolverClassical(
                self.instance,
                SolverConfig(solving=ClassicalSolvingConfig(), decompose=None),
            )
            return solver.solve()

        else:
            from qubosolver.transforms import _decompositions

            if self.config.solving_mode == "quantum":
                max_min_dist_ratio = self.config.quantum.max_min_dist_ratio
            else:
                max_min_dist_ratio = self.decomposition_config.classical_max_min_dist_ratio

            config = _decompositions.Config._from_decomposition_config(
                self.decomposition_config, max_min_dist_ratio=max_min_dist_ratio
            )
            decomposed_qubo = _decompositions.Instance(self.instance, config=config)
            solution = Solution()

            while len(decomposed_qubo._vertices_to_place) > config.decompose_stop_number:

                subqubo = _decompositions.extract_subqubo(decomposed_qubo, config, rng=rng)

                if subqubo.size == 0:
                    break
                self.number_iterations += 1

                subsolver = self._solver_factory(subqubo, self._config_subproblems)

                # only one bitstring is kept as per design choice of the
                # decomposition algorithm
                subsolution = subsolver.solve()._compute_costs(subqubo.matrix)._sort_by_cost()
                solution = _decompositions.update(decomposed_qubo, subqubo, subsolution)

            # classical resolution of last matrix
            subqubo = _decompositions.extract_subqubo(
                decomposed_qubo,
                config,
                last=True,
                rng=rng,
            )
            if subqubo.size != 0:
                lastsolver = _QuboSolverClassical(
                    subqubo,
                    SolverConfig(solving=ClassicalSolvingConfig(), decompose=None),
                )
                subsolution = lastsolver.solve()._compute_costs(subqubo.matrix)._sort_by_cost()
                solution = _decompositions.update(decomposed_qubo, subqubo, subsolution)

            self._decomposition = decomposed_qubo._decomposition

            return solution
