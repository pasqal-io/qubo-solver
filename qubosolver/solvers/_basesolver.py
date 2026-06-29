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
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, drive shaping,
    and execution of QUBO problems.

    The BaseSolver also provides a method to execute the QuantumProgram.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig = SolverConfig()):
        """
        Initialize the solver with the QUBO instance and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig | None): Configuration settings for the solver.
                Defaults to a fresh :class:`SolverConfig` if ``None``.
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
    def drive(self, embedding: Register) -> tuple:
        """
        Generate a drive for the quantum device based on the embedding.

        Args:
            embedding (Register): The atom register layout.

        Returns:
            tuple:
                - Drive: The drive schedule.
                - QUBOSolution: Initial solution from drive shaping.
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
        """Draw sequence of the `QuantumProgram` submitted.

        Args:
            drive (Drive): Drive used in program.
            embedding (Register): embedding program is defined over.
        """
        if self.config.use_quantum:
            program = solvers.quantum._quantum_program(
                embedding, drive, self.device, compiler_profile=_compiler_profile(self.config)
            )
            program.draw(compiled=True)

    def _trivial_solution(self) -> Optional[QUBOSolution]:
        """Check for trivial solutions (all-zero, all-one, or diagonal)."""
        return solvers.trivial_solution_search(self.instance)

    def _update_instance(self, instance: QUBOInstance) -> None:
        """Replace the current QUBO instance and propagate to any inner solver."""
        self.instance = instance
        # Update _solver's as well
        _solver = getattr(self, "_solver", None)
        if _solver is not None:
            assert isinstance(_solver, BaseSolver)  # nosec B101
            _solver.instance = self.instance

    def preprocess(self) -> None:
        """Apply preprocessing on instance to reduce its size."""
        if self.config.do_preprocessing:
            self._update_instance(transforms.variable_fixing.apply_recursively(self.instance))

    def post_process_fixation(self, solution: QUBOSolution) -> QUBOSolution:
        """Post-process fixations of the preprocessing and restore the original QUBO.

        Args:
            solution (QUBOSolution): Solution after preprocessing.

        Returns:
            QUBOSolution: New restored solution if preprocessing was applied.
        """
        # Means that preprocessing was not applied
        if not self.config.do_preprocessing:
            return solution

        assert isinstance(self.instance, transforms.variable_fixing.QUBOInstance)  # nosec B101
        new_solution = transforms.variable_fixing.unapply(solution, self.instance)
        self._update_instance(self.instance._parent_instance)

        return new_solution

    def post_process(self, solution: QUBOSolution) -> QUBOSolution:
        """Apply post-processing.

         Args:
            solution (QUBOSolution): Solution after preprocessing.

        Returns:
            QUBOSolution: New postprocessed solution.
        """
        if not self.config.do_postprocessing:
            return solution

        return solvers.iterative_bitflip_local_search(self.instance, solution)

    @classmethod
    def save(cls, file_like: io_utils.FileLike[bytes], solver: Self) -> None:
        """
        Save a solver instance to a file-like object.

        Note:
            This is currently a partial serialization. Only the QUBO instance,
            preprocessing/postprocessing flags, and fixed variable information
            are saved. The complete solver configuration and state are not
            fully serialized.

        Args:
            file_like (io_utils.FileLike[bytes]): A file-like object opened in binary
                write mode where the solver data will be saved. This can be a file path
                string, Path object, or any file-like object that supports binary writing.
            solver (Self): The solver instance to be saved. Must be an instance of the
                same class that this classmethod is called on.

        Returns:
            None: This method does not return a value. The solver data is written
                directly to the provided file-like object.
        """
        with io_utils.open(file_like, mode="wb") as f:
            io_utils.save(f, "?", solver.config.do_preprocessing)
            io_utils.save(f, "?", solver.config.do_postprocessing)
            if solver.config.do_preprocessing:
                assert isinstance(solver.instance, transforms.variable_fixing.QUBOInstance)
                transforms.variable_fixing.QUBOInstance.save(f, solver.instance)
            else:
                QUBOInstance.save(f, solver.instance)

    @classmethod
    def load(cls, file_like: io_utils.FileLike[bytes]) -> Self:
        """
        Load a solver instance from a file-like object.

        This method deserializes a solver that was previously saved using the save()
        method. The loaded solver is in a limited state where most methods are disabled
        except for post-processing operations. This is because the complete solver
        configuration and state cannot be fully restored from the saved data.

        Args:
            file_like (io_utils.FileLike[bytes]): A file-like object opened in binary
                read mode containing the serialized solver data. This can be a file path
                string, Path object, or any file-like object that supports binary reading.

        Returns:
            Self: A new solver instance of the same class with restored QUBO instance,
                preprocessing/postprocessing configuration, and fixed variable information.
                Note that most solver methods will be disabled and raise AttributeError
                when called, except for post_process_fixation and post_process methods.

        Note:
            The returned solver is in a limited state suitable only for post-processing
            operations. Most functionality including solving, embedding, and execution
            methods are disabled to prevent incorrect usage of an incompletely loaded solver.
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
        def disabled(*args: Any, **kwargs: Any) -> None:
            raise AttributeError(
                f"'{name}' is disabled: this method is not supported for QuboSolverQuantum loaded from a file."
            )

        return disabled
