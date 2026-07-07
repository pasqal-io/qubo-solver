from __future__ import annotations

from abc import ABC, abstractmethod

from qoolqit.execution import job

from qubosolver.types import Instance, Solution
from qubosolver.config import SolverConfig, _compiler_profile
from qubosolver import solvers, transforms


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qoolqit import Register, Drive


class BaseSolver(ABC):
    """Abstract base class for all solvers (quantum or classical).

    Concrete subclasses must implement the three abstract methods that form the
    standard QUBO-solving pipeline:

    1. :meth:`embedding` — map QUBO variables onto physical atom positions.
    2. :meth:`drive` — generate the pulse schedule that encodes the problem.
    3. :meth:`solve` — run the full pipeline and return a :class:`~qubosolver.types.Solution`.

    ``BaseSolver`` also provides shared infrastructure used by all concrete
    solvers:

    * :meth:`submit` / :meth:`execute` — compile and run a quantum program.
    * :meth:`preprocess` / :meth:`post_process_fixation` — variable-fixing
      pre- and post-processing to reduce problem size.
    * :meth:`post_process` — iterative bit-flip local search to improve solutions.
    * :meth:`draw_sequence` — visualise the compiled pulse sequence.
    * :meth:`save` / :meth:`load` — partial serialisation for deferred
      post-processing.
    """

    def __init__(self, instance: Instance, config: SolverConfig = SolverConfig()):
        """Initialise the solver with a QUBO instance and configuration.

        Args:
            instance: The QUBO problem to solve.
            config: Configuration settings for the solver (backend, device,
                embedding, drive-shaping, pre/post-processing flags, etc.).
                Defaults to a default-constructed :class:`SolverConfig`.
        """
        self.instance: Instance = instance
        self.config = config

        self.backend = self.config.backend
        self.device = self.config.device

    @property
    def n_fixed_variables_preprocessing(self) -> int:
        """Number of variables fixed during preprocessing.

        Returns:
            The count of fixed variables, or 0 if no preprocessing was applied.
        """
        if isinstance(self.instance, transforms.variable_fixing.Instance):
            return self.instance.n_fixed_indices
        else:
            return 0

    @abstractmethod
    def solve(self) -> Solution:
        """
        Solve the given QUBO instance.

        Returns:
            Solution: The result of the optimization.
        """
        pass

    @abstractmethod
    def embedding(self) -> Register:
        """
        Generate or retrieve an embedding for the QUBO instance.

        Returns:
            Register: The atom register layout for the instance.
        """
        pass

    @abstractmethod
    def drive(self, embedding: Register) -> tuple[Drive, Solution]:
        """Generate a pulse drive for the quantum device based on the embedding.

        Args:
            embedding: The atom register layout produced by :meth:`embedding`.

        Returns:
            A 2-tuple of:

            * **Drive** — the pulse schedule that encodes the QUBO problem.
            * **Solution** — an initial solution produced as a by-product
              of drive shaping (may be empty if drive shaping does not sample).
        """
        pass

    def submit(
        self,
        drive: Drive,
        embedding: Register,
    ) -> job.Job:
        """
        Submit a quantum program for execution on the configured backend.

        Creates a QuantumProgram from the provided drive and embedding, compiles it
        to the target device, and submits it for execution.

        Args:
            drive (Drive): The drive schedule containing the quantum operations to execute.
            embedding (Register): The register configuration defining the qubit layout
                and connectivity for the quantum program.

        Returns:
            job.Job: A job handle for the submitted execution.
        """
        return solvers.analog_quantum_sample(
            embedding,
            drive,
            self.backend,
            self.device,
            compiler_profile=_compiler_profile(self.config),
        )

    def execute(self, drive: Drive, embedding: Register) -> Solution:
        """
        Execute the drive schedule on the backend and retrieve the solution.

        Args:
            drive (Drive): The drive schedule to execute.
            embedding (Register): The register to execute on.

        Returns:
            Solution: The solution built from execution results.
        """
        job = self.submit(drive, embedding)
        return Solution.from_results(job.results())

    def draw_sequence(self, drive: Drive, embedding: Register) -> None:
        """Draw the compiled pulse sequence of the quantum program.

        Builds the same :class:`~qoolqit.QuantumProgram` that would be
        submitted by :meth:`submit`, compiles it, and renders the compiled
        sequence inline.

        This method is a no-op when ``config.use_quantum`` is ``False``
        (i.e. classical solver mode), since no quantum program is created.

        Args:
            drive: The pulse drive schedule to visualise.
            embedding: The atom register the program is defined over.
        """
        if self.config.use_quantum:
            program = solvers.quantum._quantum_program(
                embedding, drive, self.device, compiler_profile=_compiler_profile(self.config)
            )
            program.draw(compiled=True)

    def _trivial_solution(self) -> Solution:
        """Search for a trivial solution (all-zeros, all-ones, or pure-diagonal).

        Delegates to `~qubosolver.solvers.trivial_solution_search`.

        Returns:
            A :class:`Solution`. The solution is empty if no trivial optimum is found.
        """
        return solvers.trivial_solution_search(self.instance)

    def _update_instance(self, instance: Instance) -> None:
        """Replace the active QUBO instance on this solver and any inner solver.

        If the concrete subclass exposes a ``_solver`` attribute (e.g. a
        wrapped classical or decomposition solver), its ``instance`` is also
        updated so both stay in sync.

        Args:
            instance: The new :class:`~qubosolver.types.Instance` to use.
        """
        self.instance = instance
        # Update _solver's as well
        _solver = getattr(self, "_solver", None)
        if _solver is not None:
            assert isinstance(_solver, BaseSolver)  # nosec B101
            _solver.instance = self.instance

    def preprocess(self) -> None:
        """Apply variable-fixing preprocessing to reduce the problem size.

        This method is a no-op when ``config.do_preprocessing`` is ``False``.
        """
        if self.config.do_preprocessing:
            self._update_instance(transforms.variable_fixing.apply_recursively(self.instance))

    def post_process_fixation(self, solution: Solution) -> Solution:
        """Restore fixed variables and recover a solution over the original QUBO.

        Reverses the variable-fixing applied by :meth:`preprocess`: re-inserts
        the fixed variable values into *solution*.

        Returns *solution* unchanged when ``config.do_preprocessing`` is
        ``False``.

        Args:
            solution: The solution obtained after solving the reduced instance.

        Returns:
            A new :class:`~qubosolver.types.Solution` defined over the
            full, original QUBO variables.  Returns *solution* as-is when
            preprocessing was not applied.
        """
        # Means that preprocessing was not applied
        if not self.config.do_preprocessing:
            return solution

        assert isinstance(self.instance, transforms.variable_fixing.Instance)  # nosec B101
        new_solution = transforms.variable_fixing.unapply(solution, self.instance)
        self._update_instance(self.instance._parent_instance)

        return new_solution

    def post_process(self, solution: Solution) -> Solution:
        """Improve a solution with iterative bit-flip local search.

        When ``config.do_postprocessing`` is ``True``, applies
        `~qubosolver.solvers.iterative_bitflip_local_search` to *solution*,
        which flips individual bits one at a time and accepts changes that
        reduce the QUBO cost.

        Returns *solution* unchanged when ``config.do_postprocessing`` is
        ``False``.

        Args:
            solution: The raw solution to improve, typically the output of
                :meth:`execute` or :meth:`drive`.

        Returns:
            The improved :class:`~qubosolver.types.Solution`, or the
            original *solution* if postprocessing is disabled.
        """
        if not self.config.do_postprocessing:
            return solution

        return solvers.iterative_bitflip_local_search(self.instance, solution)
