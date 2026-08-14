from __future__ import annotations

from abc import ABC, abstractmethod

from qoolqit.execution import job

from qubosolver.types import Instance, Solution
from qubosolver.solvers.config import Config as SolverConfig
from qubosolver import solvers, transforms


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qoolqit import Register, Drive


class BaseSolver(ABC):
    """Abstract base class for all solvers (quantum or classical).

    Concrete subclasses must implement the three abstract methods that form the
    standard QUBO-solving pipeline:

    1. `embedding` — map QUBO variables onto physical atom positions.
    2. `drive` — generate the pulse schedule that encodes the problem.
    3. `solve` — run the full pipeline and return a `qubosolver.types.Solution`.

    ``BaseSolver`` also provides shared infrastructure used by all concrete
    solvers:

    * `_submit` / `_execute` — compile and run a quantum program.
    * `_preprocess` / `_post_process_fixation` — variable-fixing
      pre- and post-processing to reduce problem size.
    * `_post_process` — iterative bit-flip local search to improve solutions.
    * `_draw_sequence` — visualise the compiled pulse sequence.
    """

    def __init__(self, instance: Instance, config: SolverConfig = SolverConfig()):
        """Initialise the solver with a QUBO instance and configuration.

        Args:
            instance: The QUBO problem to solve.
            config: Configuration settings for the solver (backend, device,
                embedding, drive-shaping, pre/post-processing flags, etc.).
                Defaults to a default-constructed `SolverConfig`.
        """
        self.instance: Instance = instance
        self.config = config

        self.backend = self.config.backend
        self.device = self.config.device

    @abstractmethod
    def solve(self) -> Solution:
        """
        Solve the given QUBO instance.

        Returns:
            Solution: The result of the optimization.
        """
        pass

    @abstractmethod
    def _embedding(self) -> Register:
        """
        Generate or retrieve an embedding for the QUBO instance.

        Returns:
            Register: The atom register layout for the instance.
        """
        pass

    @abstractmethod
    def _drive(self, embedding: Register) -> tuple[Drive, Solution]:
        """Generate a pulse drive for the quantum device based on the embedding.

        Args:
            embedding: The atom register layout produced by `embedding`.

        Returns:
            A 2-tuple of:

            * **Drive** — the pulse schedule that encodes the QUBO problem.
            * **Solution** — an initial solution produced as a by-product
              of drive shaping (may be empty if drive shaping does not sample).
        """
        pass

    def _submit(
        self,
        drive: Drive,
        embedding: Register,
    ) -> job.Job:
        """
        Submit a quantum program for execution on the configured backend.

        Creates a QuantumProgram from the provided drive and embedding, compiles it
        to the target device, and submits it for execution.

        Args:
            drive: The drive schedule containing the quantum operations to execute.
            embedding: The register configuration defining the qubit layout
                and connectivity for the quantum program.

        Returns:
            A job handle for the submitted execution.
        """
        return solvers.analog_quantum_sampling(
            embedding,
            drive,
            self.backend,
            self.device,
            default_sequence_duration=self.config.drive_shaping.default_sequence_duration,
        )

    def _execute(self, drive: Drive, embedding: Register) -> Solution:
        """
        Execute the drive schedule on the backend and retrieve the solution.

        Args:
            drive (Drive): The drive schedule to execute.
            embedding (Register): The register to execute on.

        Returns:
            Solution: The solution built from execution results.
        """
        job = self._submit(drive, embedding)
        return Solution.from_results(job.results())

    def _draw_sequence(self, drive: Drive, embedding: Register) -> None:
        """Draw the compiled pulse sequence of the quantum program.

        Builds the same `qoolqit.QuantumProgram` that would be
        submitted by `_submit`, compiles it, and renders the compiled
        sequence inline.

        This method is a no-op when ``config.use_quantum`` is ``False``
        (i.e. classical solver mode), since no quantum program is created.

        Args:
            drive: The pulse drive schedule to visualise.
            embedding: The atom register the program is defined over.
        """
        if self.config.use_quantum:
            program = solvers.quantum._quantum_program(
                embedding,
                drive,
                self.device,
                default_sequence_duration=self.config.drive_shaping.default_sequence_duration,
            )
            program.draw(compiled=True)

    def _trivial_solution(self) -> Solution:
        """Search for a trivial solution (all-zeros, all-ones, or pure-diagonal).

        Delegates to `qubosolver.solvers.trivial_solution_search`.

        Returns:
            A `Solution`. The solution is empty if no trivial optimum is found.
        """
        return solvers.trivial_solution_search(self.instance)

    def _update_instance(self, instance: Instance) -> None:
        """Replace the active QUBO instance on this solver and any inner solver.

        If the concrete subclass exposes a ``_solver`` attribute (e.g. a
        wrapped classical or decomposition solver), its ``instance`` is also
        updated so both stay in sync.

        Args:
            instance: The new `qubosolver.types.Instance` to use.
        """
        self.instance = instance
        # Update _solver's as well
        _solver = getattr(self, "_solver", None)
        if _solver is not None:
            assert isinstance(_solver, BaseSolver)  # nosec B101
            _solver.instance = self.instance

    def _preprocess(self) -> None:
        """Apply preprocessing to reduce the problem size and handle negative interactions.

        Runs variable-fixing first, then GLPK bit-flip preprocessing
        to remove negative off-diagonal coefficients.

        This method is a no-op when ``config.do_preprocessing`` is ``False``.
        """
        if not self.config.do_preprocessing:
            return

        instance: Instance = transforms.variable_fixing.apply_recursively(self.instance)
        instance = transforms.negative_bitflip.apply(instance)
        assert isinstance(instance, transforms.negative_bitflip.Instance)
        self._update_instance(instance)

    def _post_process_fixation(self, solution: Solution) -> Solution:
        """Restore fixed variables and recover a solution over the original QUBO.

        Reverses the preprocessing applied by [`_preprocess`][]: first undoes any
        bit flips, then re-inserts the fixed variable values into *solution*.

        Returns *solution* unchanged when ``config.do_preprocessing`` is
        ``False``.

        Args:
            solution: The solution obtained after solving the reduced instance.

        Returns:
            A new [`qubosolver.Solution`][] defined over the
            full, original QUBO variables.  Returns *solution* as-is when
            preprocessing was not applied.
        """
        # Means that preprocessing was not applied
        if not self.config.do_preprocessing:
            return solution

        # Unwind the preprocessing layers in reverse: bit flips, then
        # variable fixing. Bit-flip and fixing each remap the solution.
        assert isinstance(self.instance, transforms.negative_bitflip.Instance)  # nosec B101
        flipped_instance = self.instance
        solution = transforms.negative_bitflip.lift(solution, flipped_instance)
        self._update_instance(flipped_instance._parent_instance)

        assert isinstance(self.instance, transforms.variable_fixing.Instance)  # nosec B101
        new_solution = transforms.variable_fixing.lift(solution, self.instance)
        self._update_instance(self.instance._parent_instance)

        return new_solution

    def _post_process(self, solution: Solution) -> Solution:
        """Improve a solution with iterative bit-flip local search.

        When ``config.do_postprocessing`` is ``True``, applies
        [`qubosolver.solvers.iterative_bitflip_local_search`][] to *solution*,
        which flips individual bits one at a time and accepts changes that
        reduce the QUBO cost.

        Returns *solution* unchanged when ``config.do_postprocessing`` is
        ``False``.

        Args:
            solution: The raw solution to improve, typically the output of
                `_execute` or `_drive`.

        Returns:
            The improved [`qubosolver.Solution`][], or the
            original *solution* if postprocessing is disabled.
        """
        if not self.config.do_postprocessing:
            return solution

        return solvers.iterative_bitflip_local_search(self.instance, solution)
