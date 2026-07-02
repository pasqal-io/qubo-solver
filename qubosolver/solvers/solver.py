"""Concrete QUBO solver implementations built on top of :class:`BaseSolver`.

This module provides three internal solver classes and the public
:class:`QuboSolver` dispatcher:

* :class:`QuboSolver` — public entry point.  Inspects
  :class:`~qubosolver.config.SolverConfig` and instantiates one of the three
  solvers below.
* :class:`_QuboSolverQuantum` — runs the full quantum pipeline: embedding →
  drive shaping → analog quantum sampling → postprocessing.
* :class:`_QuboSolverClassical` — delegates to a classical optimiser
  (CPLEX, Simulated Annealing, Tabu Search, …) via
  :func:`~qubosolver.solvers.get_classical_solver`.
* :class:`_DecomposeQuboSolver` — recursively decomposes a large QUBO into
  device-sized subproblems, solves each with a sub-solver, and merges the
  partial solutions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import qoolqit

from qubosolver.types import QUBOSolution, QUBOInstance, random
from qubosolver.config import DecompositionConfig, SolverConfig
from ._basesolver import BaseSolver
from ._classical_solver import get_classical_solver
from qubosolver.embedding._embedder import _get_embedder
from qubosolver.drive_shaping._drive_shaper import _get_drive_shaper


class QuboSolver(BaseSolver):
    """Public QUBO solver dispatcher.

    Inspects :class:`~qubosolver.config.SolverConfig` at construction time
    and selects the appropriate inner solver:

    * ``config.decompose`` set → :class:`_DecomposeQuboSolver` (with
      :class:`_QuboSolverQuantum` or :class:`_QuboSolverClassical` as the
      sub-solver factory, depending on ``config.use_quantum``).
    * ``config.use_quantum=True`` → :class:`_QuboSolverQuantum`.
    * ``config.use_quantum=False`` → :class:`_QuboSolverClassical`.

    All public methods delegate directly to the selected inner solver.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
        """Initialise and select the appropriate inner solver.

        Args:
            instance: The QUBO problem to solve.
            config: Solver configuration controlling which inner solver is
                selected and how it behaves.
        """
        super().__init__(instance, config)
        self._solver: BaseSolver

        if config is None:
            self._solver = _QuboSolverClassical(instance, self.config)
        else:
            if config.decompose:
                if self.config.use_quantum:
                    solver_factory: type[BaseSolver] = _QuboSolverQuantum
                else:
                    solver_factory = _QuboSolverClassical
                self._solver = _DecomposeQuboSolver(instance, self.config, solver_factory)

            elif config.use_quantum:
                self._solver = _QuboSolverQuantum(instance, config)
            else:
                self._solver = _QuboSolverClassical(instance, config)

    def embedding(self) -> qoolqit.Register:
        """Delegate embedding generation to the inner solver.

        Returns:
            The :class:`~qoolqit.Register` produced by the inner solver.
        """
        return self._solver.embedding()

    def drive(self, embedding: qoolqit.Register) -> tuple[qoolqit.Drive, QUBOSolution]:
        """Delegate drive generation to the inner solver.

        Args:
            embedding: The register layout produced by :meth:`embedding`.

        Returns:
            A ``(Drive, QUBOSolution)`` tuple from the inner solver.
        """
        return self._solver.drive(embedding)

    def solve(self) -> QUBOSolution:
        """Solve the QUBO instance by delegating to the selected inner solver.

        Returns:
            The :class:`~qubosolver.types.QUBOSolution` produced by the inner solver.
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

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        """Initialise the quantum solver.

        Args:
            instance: The QUBO problem to solve.  Raises immediately if any
                off-diagonal coefficient is negative or the problem exceeds
                80 variables.
            config: Solver settings (backend, device, embedding, drive
                shaping, etc.).  Defaults to
                ``SolverConfig(use_quantum=True)`` when ``None``.

        Raises:
            ValueError: If any off-diagonal coefficient in ``instance.matrix``
                is negative.
            ValueError: If ``instance.size > 80``.
        """
        super().__init__(instance, config or SolverConfig(use_quantum=True))

        if (instance.matrix[~torch.eye(*instance.matrix.shape, dtype=torch.bool)] < 0).any():
            raise ValueError("Quantum solver does not handle off-diagonal negative coefficients")

        self._check_size_limit()

        self.backend = self.config.backend
        self.embedder = _get_embedder(self.instance, self.config, self.backend)
        self.drive_shaper = _get_drive_shaper(self.instance, self.config, self.backend)

        self._register: qoolqit.Register | None = None
        self._drive: qoolqit.Drive | None = None

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

    def embedding(self) -> qoolqit.Register:
        """Embed QUBO variables onto physical atom positions.

        Calls the configured embedder (BLaDE or Greedy) and caches the
        result in ``self._register``.

        Returns:
            The atom :class:`~qoolqit.Register` layout for the current
            instance.
        """
        self.embedder.instance = self.instance
        self._register = self.embedder.embed()
        return self._register

    def drive(self, embedding: qoolqit.Register) -> tuple[qoolqit.Drive, QUBOSolution]:
        """Generate the pulse drive schedule for the given embedding.

        Calls the configured drive shaper (heuristic or optimised) and
        caches the drive in ``self._drive``.

        Args:
            embedding: The atom register layout produced by :meth:`embedding`.

        Returns:
            A 2-tuple of:

            * :class:`~qoolqit.Drive` — the pulse schedule for quantum
              execution.
            * :class:`~qubosolver.types.QUBOSolution` — an initial solution
              produced as a by-product of drive shaping (may be empty for the
              heuristic shaper).
        """
        self.drive_shaper.instance = self.instance
        drive, qubo_solution = self.drive_shaper.generate(embedding)

        self._drive = drive
        return drive, qubo_solution

    def solve(self) -> QUBOSolution:
        """Execute the full quantum pipeline and return the best solution.

        Returns:
            The final :class:`~qubosolver.types.QUBOSolution`, sorted by
            ascending cost with probabilities computed.
        """
        # 1) try trivial else delegate to quantum solver
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial:
                return trivial
        self._check_size_limit()

        # 2) Apply preprocessing if requested
        self.preprocess()

        embedding = self.embedding()

        drive, solution = self.drive(embedding)

        if not solution or self.config.drive_shaping.optimized_re_execute_opt_drive:
            solution = self.execute(drive, embedding)

        # Post-process fixations of the preprocessing and restore the original QUBO
        solution = self.post_process_fixation(solution)
        solution = self.post_process(solution)

        solution.compute_costs(self.instance.matrix).sort_by_cost().compute_probabilities()

        return solution


class _QuboSolverClassical(BaseSolver):
    """Classical QUBO solver.

    Delegates solving to the classical optimiser selected by
    :func:`~qubosolver.solvers.get_classical_solver` (CPLEX, Simulated
    Annealing, Tabu Search, or Random).

    Embedding and drive generation are not meaningful for classical solvers;
    both methods are no-ops that return ``None``.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
        super().__init__(instance, config)

    def embedding(self) -> qoolqit.Register:
        """No-op — classical solvers do not require an atom register.

        Returns:
            ``None``.
        """
        return  # type: ignore[return-value]

    def drive(self, embedding: qoolqit.Register) -> tuple:
        """No-op — classical solvers do not use a pulse drive.

        Returns:
            ``None``.
        """
        return  # type: ignore[return-value]

    def solve(self) -> QUBOSolution:
        """Solve the QUBO using the configured classical solver.

        Returns:
            The final :class:`~qubosolver.types.QUBOSolution`.
        """
        # 1) try trivial
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial:
                return trivial

        self.preprocess()

        solution = QUBOSolution()

        if self.instance.size != 0:
            classical_solver = get_classical_solver(self.instance, self.config.classical)
            solution = (
                classical_solver.solve()
            )  # This is a reduced solution if pre-procesing is applied

        solution = self.post_process_fixation(solution)
        solution = self.post_process(solution)

        return solution


class _DecomposeQuboSolver(BaseSolver):
    """Device-aware QUBO decomposition solver.

    Treats the QUBO as a graph (variables = vertices, coefficients =
    weighted edges), iteratively extracts device-sized subgraphs using
    geometric search, solves each subproblem with a configurable
    ``solver_factory``, and merges the partial solutions back into a
    global solution.  The final tail of variables (those that fall below
    :attr:`~qubosolver.config.DecompositionConfig.decompose_stop_number`)
    is always solved classically.

    Constraints:

    * All off-diagonal QUBO coefficients must be non-negative (same
      physical constraint as :class:`_QuboSolverQuantum`).
    * Exactly **one** bitstring is returned (design choice of the
      decomposition algorithm).
    """

    def __init__(
        self,
        instance: QUBOInstance,
        config: SolverConfig | None,
        solver_factory: Callable[[QUBOInstance, SolverConfig], BaseSolver],
    ):
        """Initialise the decomposition solver.

        Args:
            instance: The QUBO problem to decompose and solve.  All
                off-diagonal coefficients must be non-negative.
            config: Solver settings (backend, device, decomposition
                parameters, etc.).  Defaults to
                ``SolverConfig(use_quantum=True)`` when ``None``.
            solver_factory: A callable (typically
                :class:`_QuboSolverQuantum` or
                :class:`_QuboSolverClassical`) that constructs the
                sub-solver used to solve each extracted subproblem.

        Raises:
            ValueError: If any off-diagonal coefficient in
                ``instance.matrix`` is negative.
        """
        if (instance.matrix[~torch.eye(*instance.matrix.shape, dtype=torch.bool)] < 0).any():
            raise ValueError("Decomposition does not handle off-diagonal negative coefficients")

        # default is a quantum solver as we apply device-dependent decomposition
        super().__init__(
            QUBOInstance(instance.matrix),
            config or SolverConfig(use_quantum=True),
        )
        self._solver_factory = solver_factory

        self.backend = self.config.backend
        self.device = self.config.device

        self.decomposition_config: DecompositionConfig = (
            self.config.decompose or DecompositionConfig()
        )

        # A cached version of `config` that we're going
        # to use for problems we do not wish to decompose.
        self._config_subproblems = SolverConfig.from_kwargs(
            **self.config.model_dump(exclude={"decompose"})
        )

        self._decomposition = [list(range(instance.size))]

    def embedding(self) -> qoolqit.Register:
        """Not supported — decomposition manages embeddings per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def drive(self, embedding: qoolqit.Register) -> tuple:
        """Not supported — decomposition manages drives per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def solve(self) -> QUBOSolution:
        """Solve the QUBO by iterative decomposition into device-sized subproblems.

        Returns:
            A :class:`~qubosolver.types.QUBOSolution` containing exactly one
            bitstring — the merged result of all subproblem solutions.
        """
        # Create a local Generator that inherits whatever seeding you did via torch.manual_seed(...) (copies the current global RNG state).
        rng = random.torch_rng().set_state(torch.get_rng_state())
        self.number_iterations = 0
        assert self.instance.size  # nosec B101
        if self.instance.size <= self.decomposition_config.decompose_stop_number:
            solver = _QuboSolverClassical(
                self.instance,
                SolverConfig(use_quantum=False, decompose=None),
            )
            return solver.solve()

        else:
            from qubosolver.transforms import _decompositions

            config = _decompositions.Config.from_decomposition_config(self.decomposition_config)
            decomposed_qubo = _decompositions.QUBOInstance(
                self.instance, self.device, config=config
            )
            solution = QUBOSolution()

            while len(decomposed_qubo._vertices_to_place) > config.decompose_stop_number:

                subqubo = _decompositions.extract_subqubo(
                    decomposed_qubo, self.device, config, rng=rng
                )

                if subqubo.size == 0:
                    break
                self.number_iterations += 1

                subsolver = self._solver_factory(subqubo, self._config_subproblems)

                # only one bitstring is kept as per design choice of the
                # decomposition algorithm
                subsolution = subsolver.solve().compute_costs(subqubo.matrix).sort_by_cost()
                solution = _decompositions.update(decomposed_qubo, subqubo, subsolution)

            # classical resolution of last matrix
            subqubo = _decompositions.extract_subqubo(
                decomposed_qubo,
                self.device,
                config,
                last=True,
                rng=rng,
            )
            if subqubo.size != 0:
                lastsolver = _QuboSolverClassical(
                    subqubo,
                    SolverConfig(use_quantum=False, decompose=None),
                )
                subsolution = lastsolver.solve().compute_costs(subqubo.matrix).sort_by_cost()
                solution = _decompositions.update(decomposed_qubo, subqubo, subsolution)

            self._decomposition = decomposed_qubo._decomposition

            return solution
