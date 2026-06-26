from __future__ import annotations

from dataclasses import dataclass
import torch
from collections.abc import Iterator
from typing_extensions import Self

from ._checks import debug_runtime_typecheck
from . import vector, vectori
from . import bitstrings as _bitstrings
from . import bitstring
from .linalg import Bitstrings, Vector, Vectori, Matrix, Bitstring

from qubosolver import _utils
from pulser.backend.results import Results


@debug_runtime_typecheck
@dataclass
class QUBOSingleSolution:
    bitstring: Bitstring
    cost: float = float("inf")
    probability: float = 0.0

    @property
    def string(self) -> str:
        return bitstring.to_string(self.bitstring)


@debug_runtime_typecheck
@dataclass
class QUBOSolution:
    """
    Represents a solution to a QUBO problem.

    Attributes:
        bitstrings (Bitstrings):
            ``int8`` tensor of shape ``(num_solutions, bitstring_length)``
            containing the bitstring solutions (0s and 1s).
        costs (Vector):
            Float tensor of shape ``(num_solutions,)`` with the cost of each bitstring.
        counts (Vectori):
            ``int64`` tensor of shape ``(num_solutions,)`` with the occurrence count
            of each bitstring. Empty (``numel() == 0``) when counts are unavailable.
        probabilities (Vector):
            Float tensor of shape ``(num_solutions,)`` with the probability of each
            bitstring. Empty (``numel() == 0``) when probabilities are unavailable.
    """

    bitstrings: Bitstrings = _bitstrings.zeros(0, 0)
    costs: Vector = vector.zeros(0)
    counts: Vectori = vectori.zeros(0)
    probabilities: Vector = vector.zeros(0)

    def empty(self) -> bool:
        """Checks whether the solution contains any bitstrings.

        Returns:
            ``True`` if the solution has no bitstrings (and asserts that
            costs, counts, and probabilities are also empty), ``False`` otherwise.
        """
        r = len(self) == 0
        if not r:
            return False
        assert self.bitstrings.numel() == 0
        assert self.costs.numel() == 0
        assert self.counts.numel() == 0
        assert self.probabilities.numel() == 0

        return True

    def __bool__(self) -> bool:
        """Returns ``True`` if the solution is non-empty."""
        return not self.empty()

    def __getitem__(self, idx: int) -> QUBOSingleSolution:
        solution = QUBOSingleSolution(self.bitstrings[idx])
        if self.costs.numel() > 0:
            solution.cost = self.costs[idx].item()
        if self.probabilities.numel() > 0:
            solution.probability = self.probabilities[idx].item()

        return solution

    def __len__(self) -> int:
        return self.bitstrings.shape[0]

    def __iter__(self) -> Iterator[QUBOSingleSolution]:
        for i in range(len(self)):
            yield self[i]

    def compute_costs(self, matrix: Matrix) -> Self:
        """
        Computes the cost for each bitstring solution.

        Args:
            matrix (Matrix): The QUBO coefficient matrix of shape ``(n, n)``.

        Returns:
            Vector: A tensor of costs for each bitstring.
        """
        dtype = matrix.dtype
        self.costs = _utils.costs.batched_quadratic_cost(self.bitstrings.to(dtype), matrix)
        return self

    def compute_probabilities(self) -> Self:
        """
        Computes the probabilities of each bitstring solution based on their counts.

        Returns:
            torch.Tensor: A tensor of probabilities for each bitstring.
        """
        total_counts = self.counts.sum().item()
        self.probabilities = (
            self.counts / total_counts if total_counts > 0 else torch.zeros_like(self.counts)
        ).to(self.costs.dtype)

        return self

    def sort_by_cost(self) -> Self:
        """
        Sorts the QUBOSolution in-place by increasing cost.

        Reorders bitstrings, costs, counts, and probabilities (if available)
        based on the ascending order of the costs.
        """

        sorted_indices = torch.argsort(self.costs)
        self.bitstrings = self.bitstrings[sorted_indices]
        self.costs = self.costs[sorted_indices]
        if self.counts.numel() > 0:
            self.counts = self.counts[sorted_indices]
        if self.probabilities.numel() > 0:
            self.probabilities = self.probabilities[sorted_indices]
        return self

    @staticmethod
    def from_results(results: Results) -> QUBOSolution:
        """Builds a :class:`QUBOSolution` from Pulser simulation results.

        Args:
            results: A :class:`~pulser.backend.results.Results` object containing
                final bitstring counts.

        Returns:
            A :class:`QUBOSolution` populated with bitstrings and counts.
        """
        counter = results.final_bitstrings
        bitstrings = torch.tensor(
            [list(map(int, list(b))) for b in list(counter.keys())], dtype=torch.int8
        )
        if bitstrings.numel() == 0:
            bitstrings = torch.empty((0, 0), dtype=torch.int8)
        counts = torch.tensor(list(map(int, list(counter.values()))), dtype=torch.int64)

        return QUBOSolution(
            bitstrings=bitstrings,
            counts=counts,
        )
