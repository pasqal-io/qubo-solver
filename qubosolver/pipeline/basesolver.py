from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from qoolqit._solvers import get_backend
from qoolqit._solvers.data import QuantumProgram

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import SolutionStatusType

from .targets import Pulse, Register


class BaseSolver(ABC):
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, pulse shaping,
    and execution of QUBO problems.

    The BaseSolver also provides a method to execute the Pulse and
    Register
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
            self.config.embedding.traps = max(self.config.embedding.traps, instance.size)

        self.backend = get_backend(self.config.backend_config)

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
    def pulse(self, embedding: Register) -> tuple:
        """
        Generate a pulse schedule for the quantum device based on the embedding.

        Args:
            embedding (dict): Embedding information.

        Returns:
            tuple:
                - Pulse schedule or related data.
                - QUBOsolution
        """
        pass

    def execute(self, pulse: Pulse, embedding: Register) -> tuple:
        """
        Execute the pulse schedule on the backend and retrieve the solution.
        # TODO: We do not currently execute using the async run.
        # We are sumbitting a single job, defined in the executor.
        # In future we need to run the async functions.

        Args:
            pulse (Pulse): The pulse schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            tuple: A tuple of (bitstrings, counts) from the execution.
        """
        if hasattr(pulse, "bitstrings") and hasattr(pulse, "counts"):
            # FIXME: Can this happend?
            bitstrings, counts = pulse.bitstrings, pulse.counts
        else:
            # If not, we need to execute the simulation
            program = QuantumProgram(
                device=self.backend.device(),
                register=embedding.register,
                pulse=pulse.pulse,
            )
            execution_result = self.backend.run(program, self.config.num_shots)
            counts = execution_result.counts
            bitstrings = list(counts.keys())

        if self.config.pulse_shaping.re_execute_opt_pulse and (
            bitstrings is None or counts is None
        ):
            program = QuantumProgram(
                device=self.backend.device(),
                register=embedding.register,
                pulse=pulse.pulse,
            )
            execution_result = self.backend.run(program, self.config.num_shots)
            counts = execution_result.counts
            bitstrings = list(counts.keys())

        return bitstrings, counts

    def _trivial_solution(self) -> Optional[QUBOSolution]:
        """
        Check for the two trivial QUBO cases:
          1) all coefficients >= 0  → solution = 0^n
          2) all coefficients <= 0  → solution = 1^n

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

        return None
