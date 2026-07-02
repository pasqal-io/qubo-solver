from __future__ import annotations

from abc import ABC, abstractmethod
import inspect

from qoolqit.execution import job

from qubosolver.types import QUBOInstance, QUBOSolution
from qubosolver.config import SolverConfig, _compiler_profile
from qubosolver._io import utils as io_utils
from qubosolver import solvers, transforms


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional, Any
    from typing_extensions import Self

    from qoolqit import Register, Drive


class BaseSolver(ABC):
    """Abstract base class for all solvers (quantum or classical).

    Concrete subclasses must implement the three abstract methods that form the
    standard QUBO-solving pipeline:

    1. :meth:`embedding` — map QUBO variables onto physical atom positions.
    2. :meth:`drive` — generate the pulse schedule that encodes the problem.
    3. :meth:`solve` — run the full pipeline and return a :class:`~qubosolver.types.QUBOSolution`.

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

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
        """Initialise the solver with a QUBO instance and configuration.

        Args:
            instance: The QUBO problem to solve.
            config: Configuration settings for the solver (backend, device,
                embedding, drive-shaping, pre/post-processing flags, etc.).
                Defaults to a default-constructed :class:`SolverConfig`.
        """
        self.instance: QUBOInstance = instance
        self.config = config

        self.backend = self.config.backend
        self.device = self.config.device

    @property
    def n_fixed_variables_preprocessing(self) -> int:
        """Number of variables fixed during preprocessing.

        Returns:
            The count of fixed variables, or 0 if no preprocessing was applied.
        """
        if isinstance(self.instance, transforms.variable_fixing.QUBOInstance):
            return self.instance.n_fixed_indices
        else:
            return 0

    @abstractmethod
    def solve(self) -> QUBOSolution:
        """
        Solve the given QUBO instance.

        Returns:
            QUBOSolution: The result of the optimization.
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
    def drive(self, embedding: Register) -> tuple[Drive, QUBOSolution]:
        """Generate a pulse drive for the quantum device based on the embedding.

        Args:
            embedding: The atom register layout produced by :meth:`embedding`.

        Returns:
            A 2-tuple of:

            * **Drive** — the pulse schedule that encodes the QUBO problem.
            * **QUBOSolution** — an initial solution produced as a by-product
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

    def execute(self, drive: Drive, embedding: Register) -> QUBOSolution:
        """
        Execute the drive schedule on the backend and retrieve the solution.

        Args:
            drive (Drive): The drive schedule to execute.
            embedding (Register): The register to execute on.

        Returns:
            QUBOSolution: The solution built from execution results.
        """
        job = self.submit(drive, embedding)
        return QUBOSolution.from_results(job.results())

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

    def _trivial_solution(self) -> Optional[QUBOSolution]:
        """Search for a trivial solution (all-zeros, all-ones, or pure-diagonal).

        Delegates to :func:`~qubosolver.solvers.trivial_solution_search`.

        Returns:
            A :class:`QUBOSolution` if a trivial optimum is found, or ``None``
            if the problem has no such degenerate structure.
        """
        return solvers.trivial_solution_search(self.instance)

    def _update_instance(self, instance: QUBOInstance) -> None:
        """Replace the active QUBO instance on this solver and any inner solver.

        If the concrete subclass exposes a ``_solver`` attribute (e.g. a
        wrapped classical or decomposition solver), its ``instance`` is also
        updated so both stay in sync.

        Args:
            instance: The new :class:`~qubosolver.types.QUBOInstance` to use.
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

    def post_process_fixation(self, solution: QUBOSolution) -> QUBOSolution:
        """Restore fixed variables and recover a solution over the original QUBO.

        Reverses the variable-fixing applied by :meth:`preprocess`: re-inserts
        the fixed variable values into *solution*.

        Returns *solution* unchanged when ``config.do_preprocessing`` is
        ``False``.

        Args:
            solution: The solution obtained after solving the reduced instance.

        Returns:
            A new :class:`~qubosolver.types.QUBOSolution` defined over the
            full, original QUBO variables.  Returns *solution* as-is when
            preprocessing was not applied.
        """
        # Means that preprocessing was not applied
        if not self.config.do_preprocessing:
            return solution

        assert isinstance(self.instance, transforms.variable_fixing.QUBOInstance)  # nosec B101
        new_solution = transforms.variable_fixing.unapply(solution, self.instance)
        self._update_instance(self.instance._parent_instance)

        return new_solution

    def post_process(self, solution: QUBOSolution) -> QUBOSolution:
        """Improve a solution with iterative bit-flip local search.

        When ``config.do_postprocessing`` is ``True``, applies
        :func:`~qubosolver.solvers.iterative_bitflip_local_search` to *solution*,
        which flips individual bits one at a time and accepts changes that
        reduce the QUBO cost.

        Returns *solution* unchanged when ``config.do_postprocessing`` is
        ``False``.

        Args:
            solution: The raw solution to improve, typically the output of
                :meth:`execute` or :meth:`drive`.

        Returns:
            The improved :class:`~qubosolver.types.QUBOSolution`, or the
            original *solution* if postprocessing is disabled.
        """
        if not self.config.do_postprocessing:
            return solution

        return solvers.iterative_bitflip_local_search(self.instance, solution)

    @classmethod
    def save(cls, file_like: io_utils.FileLike[bytes], solver: Self) -> None:
        """Save a solver instance to a file-like object.

        Serialises only the data required for deferred post-processing:

        * ``do_preprocessing`` and ``do_postprocessing`` flags.
        * The QUBO instance — either the reduced
          :class:`~qubosolver.transforms.variable_fixing.QUBOInstance` (when
          preprocessing was applied) or the original
          :class:`~qubosolver.types.QUBOInstance`.

        .. note::
            This is a **partial** serialisation.  The full solver configuration
            (backend, device, drive-shaping parameters, etc.) is **not** saved.
            A solver loaded with :meth:`load` supports only
            :meth:`post_process_fixation` and :meth:`post_process`.

        Args:
            file_like: Destination for the serialised data.  Accepts a file
                path string, a :class:`pathlib.Path`, or any binary-writable
                file-like object.
            solver: The solver instance to serialise.
        """
        with io_utils.open(file_like, mode="wb") as f:
            io_utils.save(f, "?", solver.config.do_preprocessing)
            io_utils.save(f, "?", solver.config.do_postprocessing)
            if solver.config.do_preprocessing:
                assert isinstance(
                    solver.instance, transforms.variable_fixing.QUBOInstance
                )  # nosec B101
                transforms.variable_fixing.QUBOInstance.save(f, solver.instance)
            else:
                QUBOInstance.save(f, solver.instance)

    @classmethod
    def load(cls, file_like: io_utils.FileLike[bytes]) -> Self:
        """Load a solver instance from a file-like object.

        Deserialises a solver previously saved with :meth:`save` and returns it
        in a **restricted** state: all methods except
        :meth:`post_process_fixation`, :meth:`post_process`, and
        :meth:`_update_instance` are replaced with a stub that raises
        :exc:`AttributeError` on call.  This prevents accidental use of methods
        that require a fully initialised backend or device configuration.

        Args:
            file_like: Source of the serialised data.  Accepts a file path
                string, a :class:`pathlib.Path`, or any binary-readable
                file-like object.

        Returns:
            A new solver of the same concrete class with the QUBO instance and
            pre/post-processing flags restored, ready for deferred
            post-processing only.
        """
        with io_utils.open(file_like, mode="rb") as f:
            do_preprocessing: bool = io_utils.load(f, "?")
            do_postprocessing: bool = io_utils.load(f, "?")
            instance = (
                transforms.variable_fixing.QUBOInstance.load(f)
                if do_preprocessing
                else QUBOInstance.load(f)
            )

        config = SolverConfig(
            do_preprocessing=do_preprocessing, do_postprocessing=do_postprocessing
        )
        solver = cls(instance, config)

        # Solver is incompletely loaded, most functions are unvailable
        for name, _ in inspect.getmembers(solver, predicate=inspect.ismethod):
            if not name.startswith("__") and name not in (
                "post_process_fixation",
                "post_process",
                "_disabled_method",
                "_update_instance",
            ):
                setattr(solver, name, solver._disabled_method(name))

        return solver

    def _disabled_method(self, name: str) -> Callable[..., None]:
        """Return a stub callable that raises :exc:`AttributeError` when invoked.

        Used by :meth:`load` to disable all methods that cannot be correctly
        executed on an incompletely deserialised solver.

        Args:
            name: The original method name, embedded in the error message to
                aid debugging.

        Returns:
            A zero-argument-agnostic callable that always raises
            :exc:`AttributeError`.
        """
        def disabled(*args: Any, **kwargs: Any) -> None:
            raise AttributeError(
                f"'{name}' is disabled: this method is not supported for QuboSolverQuantum loaded from a file."
            )

        return disabled
