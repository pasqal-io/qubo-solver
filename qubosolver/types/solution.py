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
    """A single candidate solution extracted from a :class:`QUBOSolution`.

    Instances are normally obtained via :meth:`QUBOSolution.__getitem__` rather
    than constructed directly.

    Attributes:
        bitstring (Bitstring): Binary vector of shape ``(n,)`` with values in
            ``{0, 1}`` (``int8``).
        cost (float): Objective value ``x^T Q x``.  Defaults to ``+inf`` when
            costs have not been computed yet.
        probability (float): Sampling probability of this bitstring.  Defaults
            to ``0.0`` when probabilities are unavailable.
    """

    bitstring: Bitstring
    cost: float = float("inf")
    probability: float = 0.0

    @property
    def string(self) -> str:
        """The bitstring represented as a plain ``"0"``/``"1"`` character string."""
        return bitstring.to_string(self.bitstring)


@debug_runtime_typecheck
@dataclass
class QUBOSolution:
    """A collection of candidate solutions for a QUBO problem.

    Stores all bitstrings returned by a solver together with their associated
    metadata (costs, sample counts, probabilities).  Fields whose data is not
    yet available are represented as zero-element tensors (``numel() == 0``).

    Use :meth:`compute_costs` and :meth:`compute_probabilities` to populate
    derived fields after construction, and :meth:`sort_by_cost` to rank
    candidates for analysis.

    Attributes:
        bitstrings (Bitstrings):
            ``int8`` tensor of shape ``(num_solutions, n)`` containing candidate
            binary vectors (values in ``{0, 1}``).
        costs (Vector):
            Float tensor of shape ``(num_solutions,)`` with the QUBO objective
            ``x^T Q x`` for each bitstring.  Empty until :meth:`compute_costs`
            is called.
        counts (Vectori):
            ``int64`` tensor of shape ``(num_solutions,)`` with the number of
            times each bitstring was sampled.  Empty when the solver does not
            produce counts (e.g. exact / classical solvers).
        probabilities (Vector):
            Float tensor of shape ``(num_solutions,)`` with the empirical
            sampling probability of each bitstring.  Empty until
            :meth:`compute_probabilities` is called.
    """

    bitstrings: Bitstrings = _bitstrings.zeros(0, 0)
    costs: Vector = vector.zeros(0)
    counts: Vectori = vectori.zeros(0)
    probabilities: Vector = vector.zeros(0)

    def empty(self) -> bool:
        """Return ``True`` when the solution contains no bitstrings.

        When the solution *is* empty, also asserts (in debug / runtime-check
        builds) that all other tensors — ``costs``, ``counts``, and
        ``probabilities`` — are also empty, enforcing internal consistency.

        Returns:
            bool: ``True`` if ``len(self) == 0``, ``False`` otherwise.
        """
        r = len(self) == 0
        if not r:
            return False
        assert self.bitstrings.numel() == 0  # nosec B101
        assert self.costs.numel() == 0  # nosec B101
        assert self.counts.numel() == 0  # nosec B101
        assert self.probabilities.numel() == 0  # nosec B101

        return True

    def __bool__(self) -> bool:
        """Return ``True`` if the solution is non-empty (contains at least one bitstring)."""
        return not self.empty()

    def __getitem__(self, idx: int) -> QUBOSingleSolution:
        """Return the candidate at position *idx* as a :class:`QUBOSingleSolution`.

        Cost and probability are copied only when their respective tensors are
        non-empty; otherwise the :class:`QUBOSingleSolution` defaults
        (``cost=inf``, ``probability=0.0``) are kept.

        Args:
            idx (int): Zero-based index into the ``num_solutions`` axis.

        Returns:
            QUBOSingleSolution: Snapshot of the candidate at *idx*.
        """
        solution = QUBOSingleSolution(self.bitstrings[idx])
        if self.costs.numel() > 0:
            solution.cost = self.costs[idx].item()
        if self.probabilities.numel() > 0:
            solution.probability = self.probabilities[idx].item()

        return solution

    def __len__(self) -> int:
        """Return the number of candidate solutions (``num_solutions``)."""
        return self.bitstrings.shape[0]

    def __iter__(self) -> Iterator[QUBOSingleSolution]:
        """Iterate over all candidates in index order, yielding :class:`QUBOSingleSolution` objects."""
        for i in range(len(self)):
            yield self[i]

    def compute_costs(self, matrix: Matrix) -> Self:
        """Compute and store the QUBO objective ``x^T Q x`` for every bitstring.

        Casts :attr:`bitstrings` to the dtype of *matrix* before calling the
        batched cost kernel to avoid dtype mismatches.  The result overwrites
        :attr:`costs` in-place.

        Args:
            matrix (Matrix): QUBO coefficient matrix ``Q`` of shape ``(n, n)``.

        Returns:
            Self: The same :class:`QUBOSolution` instance, allowing method chaining.
        """
        dtype = matrix.dtype
        self.costs = _utils.costs.batched_quadratic_cost(self.bitstrings.to(dtype), matrix)
        return self

    def compute_probabilities(self) -> Self:
        """Derive empirical sampling probabilities from :attr:`counts`.

        Divides each count by the total number of samples.  When the total is
        zero (no counts recorded), the probabilities tensor is set to all zeros.
        The result is cast to the same dtype as :attr:`costs` and overwrites
        :attr:`probabilities` in-place.

        Returns:
            Self: The same :class:`QUBOSolution` instance, allowing method chaining.

        Note:
            :attr:`counts` must be populated before calling this method;
            calling it on an empty :attr:`counts` tensor produces an all-zero
            probabilities tensor.
        """
        total_counts = self.counts.sum().item()
        self.probabilities = (
            self.counts / total_counts if total_counts > 0 else torch.zeros_like(self.counts)
        ).to(self.costs.dtype)

        return self

    def sort_by_cost(self) -> Self:
        """Sort all fields in-place by ascending cost.

        Reorders :attr:`bitstrings`, :attr:`costs`, and — when non-empty —
        :attr:`counts` and :attr:`probabilities`, so that the lowest-cost
        candidate appears first.

        Returns:
            Self: The same :class:`QUBOSolution` instance, allowing method chaining.

        Note:
            :attr:`costs` must be populated (via :meth:`compute_costs`) before
            calling this method; sorting by an empty tensor raises an error.
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
        """Build a :class:`QUBOSolution` from Pulser quantum-simulation results.

        Parses ``results.final_bitstrings`` — a ``dict[str, int]`` mapping
        each observed bitstring (e.g. ``"0101"``) to its sample count — and
        converts it into the tensor representation used by :class:`QUBOSolution`.

        Only :attr:`bitstrings` and :attr:`counts` are populated; call
        :meth:`compute_costs` and :meth:`compute_probabilities` afterwards to
        derive the remaining fields.

        Args:
            results (pulser.backend.results.Results): Pulser results object
                whose ``final_bitstrings`` attribute is a ``dict[str, int]``.

        Returns:
            QUBOSolution: A new solution with:

            * ``bitstrings`` — ``int8`` tensor of shape ``(num_solutions, n)``.
            * ``counts``     — ``int64`` tensor of shape ``(num_solutions,)``.
            * ``costs`` / ``probabilities`` — empty (not yet computed).

        Note:
            When ``final_bitstrings`` is empty (no samples recorded),
            ``bitstrings`` is set to a ``(0, 0)`` tensor rather than the
            default shape inferred from an empty list, avoiding shape ambiguity.
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
