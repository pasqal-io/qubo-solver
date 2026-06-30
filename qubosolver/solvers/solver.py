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
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
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
        """Delegate embedding generation to the inner solver."""
        return self._solver.embedding()

    def drive(self, embedding: qoolqit.Register) -> tuple:
        """Delegate drive generation to the inner solver."""
        return self._solver.drive(embedding)

    def solve(self) -> QUBOSolution:
        """Solve the QUBO instance by delegating to the selected inner solver.

        Returns:
            The :class:`QUBOSolution` produced by the inner solver.
        """
        return self._solver.solve()


class _QuboSolverQuantum(BaseSolver):
    """
    Quantum solver that orchestrates the solving of a QUBO problem using
    embedding, drive shaping, and quantum execution pipelines.

    Note: Negative off-diagonal coefficients are not supported.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        """
        Initialize the QuboSolver with the given problem and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig | None): Solver settings including backend and device.
                Defaults to ``SolverConfig(use_quantum=True)`` if ``None``.
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
        if self.instance.size > 80:
            raise ValueError(
                f"QUBO size {self.instance.size}×{self.instance.size}"
                + " exceeds the maximum supported size of 80×80."
            )

    def embedding(self) -> qoolqit.Register:
        """
        Generate a physical embedding (register) for the QUBO variables.

        Returns:
            qoolqit.Register: Atom layout suitable for quantum hardware.
        """
        self.embedder.instance = self.instance
        self._register = self.embedder.embed()
        return self._register

    def drive(self, embedding: qoolqit.Register) -> tuple:
        """
        Generate the drive sequence based on the given embedding.

        Args:
            embedding (qoolqit.Register): The embedded register layout.

        Returns:
            tuple:
                A tuple of
                    - qoolqit.Drive: qoolqit.Drive schedule for quantum execution.
                    - QUBOSolution: Initial solution of generated from drive shaper

        """
        self.drive_shaper.instance = self.instance
        drive, qubo_solution = self.drive_shaper.generate(embedding)

        self._drive = drive
        return drive, qubo_solution

    def solve(self) -> QUBOSolution:
        """
        Execute the full quantum pipeline: preprocess, embed, drive, execute, postprocess.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
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
    """
    Classical solver for QUBO problems.
    This implementation delegates the classical solving task to the external
    classical solver module (e.g., CPLEX, Simulated Annealing, or Tabu Search),
    as selected via the SolverConfig.

    After obtaining the raw solution, postprocessing (e.g., bit-flip local search)
    is applied.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
        super().__init__(instance, config)

    def embedding(self) -> qoolqit.Register:
        """Not applicable for classical solvers (returns ``None``)."""
        return  # type: ignore[return-value]

    def drive(self, embedding: qoolqit.Register) -> tuple:
        """Not applicable for classical solvers (returns ``None``)."""
        return  # type: ignore[return-value]

    def solve(self) -> QUBOSolution:
        """Solve the QUBO using the configured classical solver.

        Checks for trivial solutions first, then applies preprocessing,
        classical solving, and postprocessing.

        Returns:
            The :class:`QUBOSolution` produced by the classical solver pipeline.
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
    """
    Solver that leverages qubo decomposition techniques to solve
    subproblems and merge subsolutions to propose solutions of
    the original instance. Note, the QUBO instance is seen as a graph where variables
    are vertices, and coefficients represents weighted edges.

    Scope: the decomposition only accepts qubo with positive coefficients off-diagonal coefficients.

    Algorithm:
        1. Initialize global solution and vertices to place.
        2. While we can apply decomposition:
            - Select a random vertex to place using device layout.
            - Apply a geometric search to obtain a set of vertices
                that can be placed on a device to form a subproblem.
            - Solve the subproblem with a solver.
            - Update the global solution and the vertices left to place.
        3. For remaining variables, we solve classically.

    Note, only one bitstring solution is returned.
    """

    def __init__(
        self,
        instance: QUBOInstance,
        config: SolverConfig | None,
        solver_factory: Callable[[QUBOInstance, SolverConfig], BaseSolver],
    ):
        """
        Initialize the DecomposeQuboSolver with the given problem and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig): Solver settings including backend and device.
            solver_factory (Callable[[QUBOInstance, SolverConfig], BaseSolver]): solver class
                for subproblems.
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
        # This solver doesn't generate an embedding.
        raise NotImplementedError()

    def drive(self, embedding: qoolqit.Register) -> tuple:
        # This solver doesn't generate a drive.
        raise NotImplementedError()

    def solve(self) -> QUBOSolution:
        """
        Execute a solver by decomposing the instance.
        Note, the QUBO instance is seen as a graph where variables
        are vertices, and coefficients represents weighted edges.

        Algorithm:
            1. Initialize global solution and vertices to place.
            2. While we can apply decomposition:
                - Select a random vertex to place using device layout.
                - Apply a geometric search to obtain a set of vertices
                  that can be placed on a device to form a subproblem.
                - Solve the subproblem with a solver.
                - Update the global solution and the vertices left to place.
            3. For remaining variables, we solve classically.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
                Note, only one bitstring solution is returned.
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
