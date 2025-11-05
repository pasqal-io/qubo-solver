from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from qoolqit import Register, Drive, QuantumProgram

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import SolutionStatusType


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

    def execute(self, drive: Drive, embedding: Register) -> tuple:
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
        program = QuantumProgram(
            register=embedding,
            drive=drive,
        )
        program.compile_to(self.device)
        execution_result = self.backend.run(program)

        if isinstance(execution_result, tuple):
            # local emulator result
            execution_result = execution_result[-1]
            counter = execution_result.final_bitstrings
        else:
            # remote emulator result
            execution_result = execution_result[-1]
            counter = execution_result.bitstring_counts

        bitstrings = torch.tensor([list(map(int, list(b))) for b in list(counter.keys())])
        counts = torch.tensor(list(counter.values()))

        if self.config.drive_shaping.optimized_re_execute_opt_drive and (
            bitstrings.numel() == 0 or counts.numel() == 0
        ):
            program = QuantumProgram(
                register=embedding,
                drive=drive,
            )
            program.compile_to(self.device)
            execution_result = self.backend.run(program)
            if isinstance(execution_result, tuple):
                # local emulator result
                execution_result = execution_result[-1]
                counter = execution_result.final_bitstrings
            else:
                # remote emulator result
                execution_result = execution_result[-1]
                counter = execution_result.bitstring_counts
            bitstrings = torch.tensor([list(map(int, list(b))) for b in list(counter.keys())])
            counts = torch.tensor(list(counter.values()))

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
