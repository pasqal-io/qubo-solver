from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Callable, Any
from typing_extensions import Self

import torch
import json
import inspect

from qoolqit import Register, Drive, QuantumProgram

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import SolutionStatusType
from qubosolver.pipeline.fixtures import Fixtures
import qubosolver.io.utils as io_utils
from qubosolver.config import PasqalCloud

from pulser.backend import Results
from pulser.backend.remote import RemoteResults
from qoolqit.execution.backends import PulserRemoteBackend


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

        if instance.size:
            self.config.embedding.greedy_traps = max(
                self.config.embedding.greedy_traps, instance.size
            )

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

    def submit(
        self, drive: Drive, embedding: Register, wait: bool = True
    ) -> RemoteResults | Sequence[Results]:
        program = QuantumProgram(
            register=embedding,
            drive=drive,
        )
        program.compile_to(self.device)
        if isinstance(self.backend, PulserRemoteBackend):
            return self.backend.submit(program, wait)
        else:
            if not wait:
                raise RuntimeError("Async execution is not supported on Local Backends")
            return self.backend.run(program)

    @staticmethod
    def parse_results(
        results: RemoteResults | Sequence[Results],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parse the remote results from the backend.

        Returns:
            tuple: A tuple of (bitstrings, counts) from the execution.
        """
        if isinstance(results, tuple):
            # local emulator result
            counter = results[-1].final_bitstrings
        else:
            # remote emulator result
            counter = results[-1].bitstring_counts
        bitstrings = torch.tensor([list(map(int, list(b))) for b in list(counter.keys())], dtype=torch.int64)
        if bitstrings.numel() == 0:
            bitstrings = torch.empty((0,0), dtype=torch.int64)
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
        execution_results = self.submit(drive, embedding, wait=True)
        bitstrings, counts = self.parse_results(execution_results)

        if self.config.drive_shaping.optimized_re_execute_opt_drive and (
            bitstrings.numel() == 0 or counts.numel() == 0
        ):
            execution_results = self.submit(drive, embedding, wait=True)
            bitstrings, counts = self.parse_results(execution_results)

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
            program.compile_to(self.device)
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
        """Apply preprocessing on instance to reduce its size."""
        if self.config.do_preprocessing:
            # Apply preprocessing and change the solved QUBO by the reduced one
            self.fixtures.preprocess()
            if (
                self.fixtures.reduced_qubo.coefficients is not None
                and len(self.fixtures.reduced_qubo.coefficients) > 0
                and self.fixtures.n_fixed_variables < self.instance.size
            ):

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

    @staticmethod
    def get_results(
        batch_id: str, connection: PasqalCloud, check: bool = True
    ) -> RemoteResults | None:
        results = RemoteResults(batch_id, connection)
        if not check:
            return results

        status = results.get_batch_status().name

        if status == "DONE":
            return results
        if status in ["RUNNING", "PENDING"]:
            return None

        raise RuntimeError(f"Batch failed with status {status}")
