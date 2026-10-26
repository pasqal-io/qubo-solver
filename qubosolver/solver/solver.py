"""Concrete QUBO solver implementations built on top of [`BaseSolver`][].

This module provides two internal solver classes and the public
[`Solver`][qubosolver.Solver]:

* [`Solver`][qubosolver.Solver] — public entry point.  Its concrete solving
  strategy (quantum or classical) is chosen from [`Config`][] at
  construction time.
* `_QuboSolverQuantum` — runs the full quantum pipeline: embedding →
  drive shaping → analog quantum sampling → postprocessing.
* `_QuboSolverClassical` — solves using a classical optimizer
  (CPLEX, Simulated Annealing, Tabu Search, …) via
  `~qubosolver.solver.get_classical_solver`.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import torch

from qubosolver import Instance, Solution, embedding, torch_rng, transforms
from qubosolver._checks import TYPE_CHECKING
from qubosolver.transforms import variable_fixing
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal

from ._basesolver import BaseSolver
from ._classical_solver import get_classical_solver
from ._drive_shaper import _get_drive_shaper
from ._embedder import _get_embedder
from .config import ClassicalSolvingConfig, SolverConfig
from .config.config import _DecompositionConfig

if TYPE_CHECKING:
    import qoolqit

logger = logging.getLogger(__name__)

_MAX_QUBO_SIZE = 80  # device atom-number limit


class Solver(BaseSolver):
    """A QUBO solver.

    Its concrete solving strategy (quantum or classical) is chosen from
    [`SolverConfig`][] at construction time.

    Example:
        ```python
        from qubosolver import Instance, Solver, SolverConfig

        instance = Instance(matrix)
        config = SolverConfig()
        solver = Solver(instance, config)
        solution = solver.solve()
        ```
    """

    def __init__(self, instance: Instance, config: SolverConfig | None = None) -> None:
        """Initialize the solver.

        Args:
            instance: The QUBO problem to solve.
            config: Solver configuration controlling which solving strategy
                is used and how it behaves.
        """
        config = config or SolverConfig()
        super().__init__(instance, config)
        self._solver: BaseSolver

        if self.config.solving_mode == "quantum":
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

    def __init__(self, instance: Instance, config: SolverConfig | None = None) -> None:
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
        super().__init__(instance, config or SolverConfig())

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
        if self.instance.size > _MAX_QUBO_SIZE:
            msg = (
                f"QUBO size {self.instance.size}x{self.instance.size}"
                f" exceeds the maximum supported size of {_MAX_QUBO_SIZE}x{_MAX_QUBO_SIZE}."
            )
            raise ValueError(msg)

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
        assert self.config.solving_mode == "quantum"  # nosec B101

        # 1) try trivial else delegate to quantum solver
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

        if self.instance.size == 0:
            logger.info(
                "Instance has zero variables after preprocessing; skipping quantum pipeline."
            )
            solution = Solution.zeros(0)
        else:
            embedding = self._embedding()
            drive, solution = self._drive(embedding)

            if not solution or self.config.quantum.drive_shaping.algorithm != "bayesian_search":
                solution = self._execute(drive, embedding)

        if isinstance(self.instance, transforms.zeroing.Instance):
            solution = transforms.zeroing.lift(solution, self.instance)
            self._update_instance(self.instance._parent_instance)

        # Post-process fixations of the preprocessing and restore the original QUBO
        solution = self._post_process_fixation(solution)
        return self._post_process(solution)


class _QuboSolverClassical(BaseSolver):
    """Classical QUBO solver.

    Delegates solving to the classical optimiser selected by
    `~qubosolver.solvers.get_classical_solver` (CPLEX, Simulated
    Annealing, Tabu Search, or Random).

    Embedding and drive generation are not meaningful for classical solvers;
    both methods are no-ops that return ``None``.
    """

    def __init__(self, instance: Instance, config: SolverConfig | None = None) -> None:
        super().__init__(instance, config or SolverConfig())

    def _embedding(self) -> qoolqit.Register:
        """No-op — classical solvers do not require an atom register.

        Returns:
            ``None``.
        """
        return  # type: ignore[return-value]

    def _drive(self, embedding: qoolqit.Register) -> tuple:  # noqa: ARG002 (no-op override)
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
        trivial = self._trivial_solution()
        if trivial:
            return trivial

        self._preprocess()

        solution = Solution()

        classical_solver = get_classical_solver(self.instance, self.config.classical)
        solution = classical_solver.solve()

        solution = self._post_process_fixation(solution)
        return self._post_process(solution)


class _DecomposeQuboSolver(BaseSolver):
    """Device-aware QUBO decomposition solver.

    Iteratively selects device-sized clusters of variables with the
    [`geometric`][qubosolver.embedding.geometric] partial embedder, solves the
    exact sub-instance on each cluster with a `Solver` configured from
    ``solver_config``, and fixes the cluster to the best candidate with
    [`variable_fixing`][qubosolver.transforms.variable_fixing], which folds it
    into the remaining variables (then Hansen's rule fixes what it can). The
    final tail of variables (once at most
    `_DecompositionConfig.decompose_stop_number` remain, or the embedder
    places at most `_DecompositionConfig.decompose_break_placement`) is
    always solved classically.

    Constraints:

    * All off-diagonal QUBO coefficients must be non-negative (same
      physical constraint as `_QuboSolverQuantum`).
    * Only the best candidate of each subproblem is kept, so the returned
      candidates differ only on the classically solved tail: they are the
      tail solver's candidates, lifted to the full instance.
    """

    def __init__(
        self,
        instance: Instance,
        *,
        solver_config: SolverConfig | None = None,
        decompose_config: _DecompositionConfig | None = None,
    ) -> None:
        """Initialise the decomposition solver.

        Args:
            instance: The QUBO problem to decompose and solve.  All
                off-diagonal coefficients must be non-negative.
            solver_config: Solver settings (backend, device, embedding,
                drive shaping, etc.) used for the sub-solver that solves
                each extracted subproblem. In quantum mode, its device also
                bounds the distance ratio of the partial embedding. The
                remaining tail, and instances small enough not to be
                decomposed, are solved with a default classical
                configuration instead. Defaults to a default-constructed
                `SolverConfig`.
            decompose_config: The decomposition parameters to use.
                Defaults to a default-constructed `_DecompositionConfig`.

        Raises:
            ValueError: If any off-diagonal coefficient in
                ``instance.matrix`` is negative.
        """
        if (instance.matrix[~torch.eye(*instance.matrix.shape, dtype=torch.bool)] < 0).any():
            msg = "Decomposition does not handle off-diagonal negative coefficients"
            raise ValueError(msg)

        # default is a quantum solver as we apply device-dependent decomposition
        super().__init__(
            Instance(instance.matrix),
            solver_config or SolverConfig(),
        )

        self.decomposition_config = decompose_config or _DecompositionConfig()

        # A cached version of `config` that we're going
        # to use for problems we do not wish to decompose.
        self._config_subproblems = deepcopy(self.config)

    def _embedding(self) -> qoolqit.Register:
        """Not supported — decomposition manages embeddings per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    def _drive(self, embedding: qoolqit.Register) -> tuple:
        """Not supported — decomposition manages drives per subproblem.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    def solve(self) -> Solution:
        """Solve the QUBO by iterative decomposition into device-sized subproblems.

        Returns:
            A [`Solution`][qubosolver.Solution] over the full instance: the
                candidates of the classical solver for the remaining variables,
                combined with the fixed best candidate of every subproblem.
                Instances small enough not to be decomposed are solved
                classically as a whole.

        Raises:
            RuntimeError: If a sub-solver returns no candidate.
        """
        self.number_iterations = 0
        assert self.instance.size  # nosec B101
        if self.instance.size <= self.decomposition_config.decompose_stop_number:
            solver = _QuboSolverClassical(
                self.instance,
                SolverConfig(solving=ClassicalSolvingConfig()),
            )
            return solver.solve()

        if self.config.solving_mode == "quantum":
            max_min_dist_ratio = self.config.quantum.max_min_dist_ratio
        else:
            max_min_dist_ratio = self.decomposition_config.classical_max_min_dist_ratio
        geometric_config = self.decomposition_config._to_geometric_config(
            max_min_dist_ratio=max_min_dist_ratio
        )

        # The embedder requires negative diagonal coefficients: Hansen's rule fixes the others.
        reduced_instance = variable_fixing.apply_recursively(self.instance)

        rng = torch_rng("global")
        while reduced_instance.size > self.decomposition_config.decompose_stop_number:
            _, variables = embedding.geometric.embed(
                reduced_instance, config=geometric_config, rng=rng
            )
            if len(variables) <= self.decomposition_config.decompose_break_placement:
                break

            # The subproblem is re-embedded by the configured embedder.
            sub_instance = reduced_instance.subinstance(variables)
            sub_solution = Solver(sub_instance, self._config_subproblems).solve()
            if not sub_solution:
                msg = (
                    "The sub-solver returned no candidate for a subproblem of size "
                    f"{sub_instance.size}."
                )
                raise RuntimeError(msg)
            self.number_iterations += 1

            # Fixing in two separate calls: the candidate rule alone (its indices refer to
            # `reduced_instance` as it is now), then Hansen to a fixed point, since fixing variables
            # to 1 can turn diagonal coefficients non-negative.
            rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced_instance)
            variable_fixing.apply(reduced_instance, (rule,), inplace=True)
            variable_fixing.apply_recursively(reduced_instance, inplace=True)

        # classical resolution of the remaining variables
        rest_solution = _QuboSolverClassical(
            Instance(reduced_instance.matrix.clone()),
            SolverConfig(solving=ClassicalSolvingConfig()),
        ).solve()

        return variable_fixing.lift(rest_solution, reduced_instance)
