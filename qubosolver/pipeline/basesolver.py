from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import json
import torch

from qoolqit import QuantumProgram
from qoolqit.execution.job import Job

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, compiler_profile, max_duration_ratio
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import SolutionStatusType
from qubosolver.pipeline.fixtures import Fixtures
import qubosolver.io.utils as io_utils


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Callable, Any
    from typing_extensions import Self

    from qoolqit import Register, Drive
    from pulser.backend import Results


class BaseSolver(ABC):
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, drive shaping,
    and execution of QUBO problems.

    The BaseSolver also provides a method to execute the QuantumProgram.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        """
        Initialize the solver with the QUBO instance and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig): Configuration settings for the solver.
        """
        self.instance: QUBOInstance = instance

        if config is None:
            self.config = SolverConfig()
        else:
            self.config = config

        self.backend = self.config.backend
        self.device = self.config.device

        self.fixtures = Fixtures(self.instance, self.config)
        self.n_fixed_variables_preprocessing = 0

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
            dict: Embedding information for the instance.
        """
        pass

    @abstractmethod
    def drive(self, embedding: Register) -> tuple:
        """
        Generate a drive for the quantum device based on the embedding.

        Args:
            embedding (dict): Embedding information.

        Returns:
            tuple:
                - Drive or related data.
                - QUBOsolution
        """
        pass

    def submit(self, drive: Drive, embedding: Register) -> Job:
        """Compile and submit a quantum program to the backend.

        Creates a ``QuantumProgram`` from the drive and embedding, compiles it
        to the target device using the configured compiler profile and duration
        ratio, then submits it via ``self.backend.run()``.

        Args:
            drive (Drive): Drive schedule containing the quantum operations.
            embedding (Register): Register defining the qubit layout.

        Returns:
            Job: A job handle for retrieving results.
        """
        program = QuantumProgram(
            register=embedding,
            drive=drive,
        )
        program.compile_to(
            self.device,
            profile=compiler_profile(self.config),
            device_max_duration_ratio=max_duration_ratio(self.config),
        )

        return self.backend.run(program)

    @staticmethod
    def parse_results(results: Results) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse quantum backend results into standardised tensors.

        Extracts bitstring measurements and their counts from a backend
        ``Results`` object. Handles both remote and local result formats.

        Args:
            results (Results): Execution results from the quantum backend.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``bitstrings``: shape (n_samples, n_qubits), dtype int64.
                  Empty tensor of shape (0, 0) if no measurements.
                - ``counts``: shape (n_samples,), dtype int64.
        """
        counter = results.final_bitstrings
        bitstrings = torch.tensor(
            [list(map(int, list(b))) for b in list(counter.keys())], dtype=torch.int64
        )
        if bitstrings.numel() == 0:
            bitstrings = torch.empty((0, 0), dtype=torch.int64)
        counts = torch.tensor(list(map(int, list(counter.values()))), dtype=torch.int64)
        return bitstrings, counts

    def execute(self, drive: Drive, embedding: Register) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the drive schedule on the backend and retrieve the solution.
        # TODO: We do not currently execute using the async run.
        # We are sumbitting a single job, defined in the executor.
        # In future we need to run the async functions.

        Args:
            drive (Drive): The drive schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            tuple: A tuple of (bitstrings, counts) from the execution.
        """
        job = self.submit(drive, embedding)
        bitstrings, counts = self.parse_results(job.results())

        if self.config.drive_shaping.optimized_re_execute_opt_drive and (
            bitstrings.numel() == 0 or counts.numel() == 0
        ):
            job = self.submit(drive, embedding)
            bitstrings, counts = self.parse_results(job.results())

        return bitstrings, counts

    def draw_sequence(self, drive: Drive, embedding: Register) -> None:
        """Draw sequence of the `QuantumProgram` submitted.

        Args:
            drive (Drive): Drive used in program.
            embedding (Register): embedding program is defined over.
        """
        if self.config.use_quantum:
            program = QuantumProgram(
                register=embedding,
                drive=drive,
            )
            program.compile_to(
                self.device,
                profile=compiler_profile(self.config),
                device_max_duration_ratio=max_duration_ratio(self.config),
            )
            program.draw(compiled=True)

    def _trivial_solution(self) -> Optional[QUBOSolution]:
        """
        Check for the two trivial QUBO cases:
          1) all coefficients >= 0  → solution = 0^n
          2) all coefficients <= 0  → solution = 1^n
          3) diagonal qubo,  negative coeffs gets 1, positive gets 0

        Returns:
            QUBOSolution if a trivial case applies, else None.
        """
        coeffs = self.instance.coefficients  # torch.Tensor (n, n)
        n = self.instance.size
        device, dtype = coeffs.device, coeffs.dtype

        # Case 1: all coeffs >= 0 → x = [0,...,0]
        if torch.all(coeffs >= 0):
            raw = torch.zeros(n, dtype=torch.int64, device=device)
            # always make a batch of one: shape (1, n)
            batch = raw.unsqueeze(0)
            cost = self.instance.evaluate_solution(raw)
            return QUBOSolution(
                bitstrings=batch,
                costs=torch.tensor([cost], dtype=dtype, device=device),
                solution_status=SolutionStatusType.TRIVIALZERO,
            )

        # Case 2: all coeffs <= 0 → x = [1,...,1]
        if torch.all(coeffs <= 0):
            raw = torch.ones(n, dtype=torch.int64, device=device)
            # always make a batch of one: shape (1, n)
            batch = raw.unsqueeze(0)
            cost = self.instance.evaluate_solution(raw)
            return QUBOSolution(
                bitstrings=batch,
                costs=torch.tensor([cost], dtype=dtype, device=device),
                solution_status=SolutionStatusType.TRIVIALONE,
            )

        # Case 3: diagonal cases
        # negative coeffs gets 1, positive gets 0
        diagonal = torch.diag(coeffs)
        if (torch.diag(diagonal) == coeffs).all():
            raw = (diagonal < 0).long()
            cost = self.instance.evaluate_solution(raw)
            batch = raw.unsqueeze(0)
            return QUBOSolution(
                bitstrings=batch,
                costs=torch.tensor([cost], dtype=dtype, device=device),
                solution_status=SolutionStatusType.TRIVIALDIAGONAL,
            )
        return None

    def preprocess(self) -> None:
        """Apply preprocessing on instance."""
        if self.config.do_preprocessing:
            # Apply preprocessing on the working QUBO.
            # Note: reduced_qubo is the current preprocessed QUBO.
            # It can be smaller after fixation, or same-size after bit-flip/zeroing.
            self.fixtures.preprocess()

            has_preprocessed_qubo = (
                self.fixtures.reduced_qubo.coefficients is not None
                and len(self.fixtures.reduced_qubo.coefficients) > 0
            )

            has_fixation = self.fixtures.n_fixed_variables < self.instance.size
            has_bitflip = getattr(self.fixtures, "bitflip_applied", False)
            has_zeroing = getattr(self.fixtures, "zeroing_applied", False)

            if has_preprocessed_qubo and (has_fixation or has_bitflip or has_zeroing):
                self.instance = self.fixtures.reduced_qubo
                self.n_fixed_variables_preprocessing = self.fixtures.n_fixed_variables
                
    def post_process_fixation(self, solution: QUBOSolution) -> QUBOSolution:
        """Post-process fixations of the preprocessing and restore the original QUBO.

        Args:
            solution (QUBOSolution): Solution after preprocessing.

        Returns:
            QUBOSolution: New restored solution if preprocessing was applied.
        """
        if self.config.do_preprocessing:
            solution = self.fixtures.post_process_fixation(solution)
            self.instance = self.fixtures.instance
        return solution

    def post_process(self, solution: QUBOSolution) -> QUBOSolution:
        """Apply post-processing.

         Args:
            solution (QUBOSolution): Solution after preprocessing.

        Returns:
            QUBOSolution: New postprocessed solution.
        """

        if self.config.do_postprocessing:
            solution = self.fixtures.postprocess(solution)
        return solution

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
        with io_utils.open(file_like, "wb") as f:
            if solver.config.do_preprocessing:
                QUBOInstance.save(f, solver.fixtures.instance)
            else:
                QUBOInstance.save(f, solver.instance)
            io_utils.save(f, "?", solver.config.do_preprocessing)
            io_utils.save(f, "?", solver.config.do_postprocessing)

            fixed_var_json = json.dumps(solver.fixtures.fixed_var_dict_list)
            io_utils.save_string(f, fixed_var_json)

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
        with io_utils.open(file_like, "rb") as f:
            instance = QUBOInstance.load(f)
            do_preprocessing: bool = io_utils.load(f, "?")
            do_postprocessing: bool = io_utils.load(f, "?")
            fixed_var_json = io_utils.load_string(f)

        config = SolverConfig(
            do_preprocessing=do_preprocessing, do_postprocessing=do_postprocessing
        )
        solver = cls(instance, config)

        def decode_int_keys(obj: dict) -> dict:
            return {int(k): v for k, v in obj.items()}

        solver.fixtures.fixed_var_dict_list = json.loads(
            fixed_var_json, object_hook=decode_int_keys
        )

        # Solver is incompletely loaded, most functions are unvailable
        for name, _ in inspect.getmembers(solver, predicate=inspect.ismethod):
            if not name.startswith("__") and name not in (
                "post_process_fixation",
                "post_process",
                "_disabled_method",
            ):
                setattr(solver, name, solver._disabled_method(name))

        return solver

    def _disabled_method(self, name: str) -> Callable[..., None]:
        def disabled(*args: Any, **kwargs: Any) -> None:
            raise AttributeError(
                f"'{name}' is disabled: this method is not supported for QuboSolverQuantum loaded from a file."
            )

        return disabled
