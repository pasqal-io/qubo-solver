from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import torch
from collections.abc import Iterable, Iterator
from typing_extensions import Self

from ._checks import debug_runtime_typecheck
from . import bitstring, vector, vectori
from . import bitstrings as _bitstrings
from .linalg import Bitstrings, Vector, Vectori, Matrix, Bitstring
from .instance import Instance
from qubosolver._io import utils as io_utils

from pulser.backend.results import Results

logger = logging.getLogger(__name__)


@debug_runtime_typecheck
@dataclass
class SingleSolution:
    """A single candidate solution extracted from a [Solution][].

    Instances are normally obtained via [`Solution.__getitem__`][] rather
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
    count: int = 0
    probability: float = 0.0

    @property
    def string(self) -> str:
        """The bitstring represented as a plain ``"0"``/``"1"`` character string."""
        return bitstring.to_string(self.bitstring)


@debug_runtime_typecheck
@dataclass
class Solution:
    """A collection of candidate solutions for a QUBO problem.

    Stores all bitstrings returned by a solver together with their associated
    metadata (costs, sample counts, probabilities).  Fields whose data is not
    yet available are represented as zero-element tensors (``numel() == 0``).

    Use `_compute_costs` and `_compute_probabilities` to populate
    derived fields after construction, and `_sort_by_cost` to rank
    candidates for analysis.

    Attributes:
        bitstrings (Bitstrings):
            ``int8`` tensor of shape ``(num_solutions, n)`` containing candidate
            binary vectors (values in ``{0, 1}``).
        costs (Vector):
            Float tensor of shape ``(num_solutions,)`` with the QUBO objective
            ``x^T Q x`` for each bitstring.  Empty until `_compute_costs`
            is called.
        counts (Vectori):
            ``int64`` tensor of shape ``(num_solutions,)`` with the number of
            times each bitstring was sampled.  Empty when the solver does not
            produce counts (e.g. exact / classical solvers).
        probabilities (Vector):
            Float tensor of shape ``(num_solutions,)`` with the empirical
            sampling probability of each bitstring.  Empty until
            `_compute_probabilities` is called.
    """

    bitstrings: Bitstrings = _bitstrings.zeros(0, 0)
    costs: Vector = vector.zeros(0)
    counts: Vectori = vectori.zeros(0)
    probabilities: Vector = vector.zeros(0)

    def __getitem__(self, idx: int) -> SingleSolution:
        """Return the candidate at position *idx* as a `SingleSolution`.

        Cost and probability are copied only when their respective tensors are
        non-empty; otherwise the `SingleSolution` defaults
        (``cost=inf``, ``probability=0.0``) are kept.

        Args:
            idx (int): Zero-based index into the ``num_solutions`` axis.

        Returns:
            Snapshot of the candidate at *idx*.
        """
        solution = SingleSolution(self.bitstrings[idx])
        solution.count = int(self.counts[idx].item())
        if self.costs.numel() > 0:
            solution.cost = self.costs[idx].item()
        if self.probabilities.numel() > 0:
            solution.probability = self.probabilities[idx].item()

        return solution

    def __len__(self) -> int:
        """Return the number of candidate solutions (``num_solutions``)."""
        return self.bitstrings.shape[0]

    def __iter__(self) -> Iterator[SingleSolution]:
        """Iterate over all candidates in index order, yielding `SingleSolution` objects."""
        for i in range(len(self)):
            yield self[i]

    def _compute_costs(self, matrix: Matrix) -> Self:
        """Compute and store the QUBO objective ``x^T Q x`` for every bitstring.

        Casts `bitstrings` to the dtype of *matrix* before calling the
        batched cost kernel to avoid dtype mismatches.  The result overwrites
        `costs` in-place.

        Args:
            matrix: QUBO coefficient matrix ``Q`` of shape ``(n, n)``.

        Returns:
            The same [`Solution`][] instance, allowing method chaining.
        """
        # Import here to avoid circular imports
        from qubosolver.utils import _costs

        dtype = matrix.dtype
        self.costs = _costs.batched_quadratic_cost(self.bitstrings.to(dtype), matrix)
        return self

    def _compute_probabilities(self) -> Self:
        """Derive empirical sampling probabilities from `counts`.

        Divides each count by the total number of samples.  When the total is
        zero (no counts recorded), the probabilities tensor is set to all zeros.
        The result is cast to the same dtype as `costs` and overwrites
        `probabilities` in-place.

        Returns:
            The same [`Solution`][] instance, allowing method chaining.

        Note:
            `counts` must be populated before calling this method;
            calling it on an empty `counts` tensor produces an all-zero
            probabilities tensor.
        """
        total_counts = self.counts.sum().item()
        self.probabilities = (
            self.counts / total_counts if total_counts > 0 else torch.zeros_like(self.counts)
        ).to(self.costs.dtype)

        return self

    def _sort_by_cost(self) -> Self:
        """Sort all fields in-place by ascending cost.

        Reorders `bitstrings`, `costs`, and — when non-empty —
        `counts` and `probabilities`, so that the lowest-cost
        candidate appears first.

        Returns:
            The same [`Solution`][] instance, allowing method chaining.

        Note:
            `costs` must be populated (via `_compute_costs`) before
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

    def truncate(self, k: int) -> Self:
        """Keep only the first *k* candidates in-place.

        Slices `bitstrings`, `costs`, `counts`, and `probabilities`
        (whichever are non-empty) down to their first *k* rows. Does not
        sort or deduplicate first; combine with `sort_by_cost` /
        `deduplicate` as needed, e.g. ``solution.sort_by_cost().truncate(1)``
        to keep the best candidate.

        Args:
            k: Number of candidates to keep. When ``k >= len(self)``, this
                is a no-op.

        Returns:
            The same [`Solution`][] instance, allowing method chaining.

        Raises:
            ValueError: If `probabilities` is non-empty while
                `counts` is empty.

        Note:
            When both are populated, `probabilities` are recomputed
            from the truncated `counts` (via `compute_probabilities`)
            rather than merely sliced, so they still sum to 1.
        """
        if self.probabilities.numel() > 0 and self.counts.numel() == 0:
            raise ValueError("Solution.truncate() requires counts when probabilities is populated")

        self.bitstrings = self.bitstrings[:k]
        if self.costs.numel() > 0:
            self.costs = self.costs[:k]
        if self.counts.numel() > 0:
            self.counts = self.counts[:k]
            if self.probabilities.numel() > 0:
                self._compute_probabilities()
        return self

    def deduplicate(self) -> Self:
        """Collapse duplicate bitstrings in-place, summing their counts.

        Rows sharing the same bitstring are merged into a single row:
        `counts` are summed, the minimum `cost` is kept, and
        `probabilities` are recomputed from the new totals. The result
        is sorted by cost.

        When `counts` is empty (not yet populated), count-summing and
        probability recomputation are skipped, and `counts` /
        `probabilities` remain empty on the result. Likewise, when
        `costs` is empty, cost reduction and cost-based sorting are
        skipped, and `costs` remains empty on the result.

        Returns:
            The same [`Solution`][] instance, allowing method chaining.

        Note:
            When several rows share a bitstring, the minimum of their
            `costs` is kept. This is only meaningful if those costs
            were all computed from the same QUBO instance (same
            `matrix` passed to `compute_costs`); the caller is
            responsible for ensuring that, since this method does not
            verify it.
        """
        if not self:
            return self

        unique_bitstrings, inverse = self.bitstrings.unique(dim=0, return_inverse=True)
        n = unique_bitstrings.shape[0]
        self.bitstrings = unique_bitstrings

        if self.counts.numel() > 0:
            self.counts = vectori.zeros(n).scatter_reduce(
                dim=0, index=inverse, src=self.counts, reduce="sum", include_self=False
            )

        if self.costs.numel() > 0:
            self.costs = vector.zeros(n).scatter_reduce(
                dim=0, index=inverse, src=self.costs, reduce="amin", include_self=False
            )
            self._sort_by_cost()

        if self.counts.numel() > 0:
            self._compute_probabilities()

        return self

    @staticmethod
    def concat(solutions: Iterable[Solution], *, unit_counts: bool = False) -> Solution:
        """Concatenate several solutions into a new one, without deduplication.

        Concatenates `bitstrings`, `costs`, `counts`, and
        `probabilities` from every solution in *solutions*. Duplicate
        bitstrings, if any, are kept as separate rows — call
        `deduplicate` on the result to collapse them:

        ```python
        merged = Solution.concat([a, b]).deduplicate()
        ```

        Args:
            solutions: Solutions to concatenate. Empty solutions (no
                bitstrings) are skipped. Among the remaining solutions,
                each of `costs`, `counts`, and `probabilities` must be
                either populated on all of them or empty on all of them
                — mixing populated and empty for the same field raises
                `ValueError`.
            unit_counts: When ``True``, set `counts` to ``1`` for every
                concatenated candidate instead of concatenating their
                original counts — useful when each candidate should count
                as a single vote once merged, e.g.
                ``Solution.concat(solutions, unit_counts=True).deduplicate()``.
                `probabilities`, if populated, are recomputed from the
                resulting `counts` rather than concatenated, so they
                still sum to 1.

        Returns:
            A new [`Solution`][] containing every candidate from every
            solution, or an empty [`Solution`][] if *solutions* is empty
            or contains only empty solutions.
        """
        non_empty = [solution for solution in solutions if solution]
        if not non_empty:
            return Solution()

        for field in ("costs", "counts", "probabilities"):
            populated = [getattr(s, field).numel() > 0 for s in non_empty]
            if any(populated) and not all(populated):
                raise ValueError(
                    f"Solution.concat() requires {field} to be either populated on "
                    f"every solution or empty on every solution, not a mix of both"
                )

        bitstrings = torch.cat([s.bitstrings for s in non_empty], dim=0)
        probabilities_populated = non_empty[0].probabilities.numel() > 0
        if unit_counts:
            counts = vectori.zeros(bitstrings.shape[0]).fill_(1)
        else:
            counts = torch.cat([s.counts for s in non_empty], dim=0)

        result = Solution(
            bitstrings=bitstrings,
            costs=torch.cat([s.costs for s in non_empty], dim=0),
            counts=counts,
        )
        if probabilities_populated:
            result._compute_probabilities()
        return result

    @staticmethod
    def from_results(results: Results, *, instance: Instance = Instance()) -> Solution:
        """Build a [`Solution`][] from Pulser quantum-simulation results.

        Parses ``results.final_bitstrings`` — a ``dict[str, int]`` mapping
        each observed bitstring (e.g. ``"0101"``) to its sample count — and
        converts it into the tensor representation used by [`Solution`][].

        Only `bitstrings` and `counts` are populated; call
        `_compute_costs` and `_compute_probabilities` afterwards to
        derive the remaining fields.

        Args:
            results (pulser.backend.results.Results): Pulser results object
                whose ``final_bitstrings`` attribute is a ``dict[str, int]``.

        Returns:
            A new solution with:

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

        solution = Solution(
            bitstrings=bitstrings,
            counts=counts,
        )

        if instance.size != 0:
            solution._compute_costs(instance.matrix)._sort_by_cost()._compute_probabilities()

        return solution

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], solution: Solution) -> None:
        """Serialise *solution* to *file_like* using `torch.save`.

        All four tensor fields are packed into a single dict and written into
        an internal `io.BytesIO` buffer, then flushed to *file_like* using
        `io_utils.save_sized_buffer`, which prefixes the payload with its
        byte length. This framing allows multiple objects to be stored
        contiguously in the same stream.

        Args:
            file_like (io_utils.FileLike[bytes]):
                Destination — a file path (`str` or `os.PathLike`),
                or a binary-writable `typing.IO` stream.
            solution (Solution):
                The solution to serialise.
        """
        with io_utils.open(file_like, "wb") as f:
            buffer = io.BytesIO()
            torch.save(
                {
                    "bitstrings": solution.bitstrings,
                    "costs": solution.costs,
                    "counts": solution.counts,
                    "probabilities": solution.probabilities,
                },
                buffer,
            )
            io_utils.save_sized_buffer(f, buffer.getbuffer())

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> Solution:
        """Deserialise a `Solution` previously saved with `save`.

        Reads a length-prefixed byte block from *file_like* into a dedicated
        `io.BytesIO` buffer before calling `torch.load`. The isolated
        buffer prevents `torch.load` from over-consuming the source stream
        when multiple objects are packed together.

        Args:
            file_like (io_utils.FileLike[bytes]):
                Source — a file path (`str` or `os.PathLike`),
                or a binary-readable `typing.IO` stream. Must contain
                data written by `save`.

        Returns:
            A new solution with the tensor fields deserialised from *file_like*.

        Note:
            `torch.load` is called with `weights_only=True` to prevent
            arbitrary code execution from untrusted checkpoint files.
        """
        with io_utils.open(file_like, "rb") as f:
            # torch.load might consume too much of the src buffer.
            #  Use a dedicated limited buffer
            buffer = io.BytesIO(io_utils.load_sized_buffer(f))
            data = torch.load(buffer, weights_only=True)

        return Solution(
            bitstrings=data["bitstrings"],
            costs=data["costs"],
            counts=data["counts"],
            probabilities=data["probabilities"],
        )

    def check_consistency(self, *, instance: Instance | None, throw: bool = False) -> bool:
        """Check internal consistency of this solution against a QUBO instance.

        Recomputes costs from `bitstrings` and `instance.matrix` and checks
        for duplicate rows, so this can be slow on large solutions — prefer
        calling it in tests / debugging rather than on every solver result.

        Verifies that:

        * `bitstrings` has ``instance.size`` columns.
        * `costs`, `counts`, and `probabilities` each have exactly
          `len(self)` elements (i.e. none of them is empty).
        * `costs` matches ``x^T Q x`` for every bitstring, computed from
          ``instance.matrix``.
        * `costs` is sorted in non-decreasing order.
        * `probabilities` matches `counts` normalised by their sum.
        * `counts` are strictly positive integers.
        * `bitstrings` contains no duplicate rows.
        * `bitstrings` entries are all ``0`` or ``1``.

        Args:
            instance: The QUBO instance this solution is expected to solve.
            throw: When ``True``, raise an `AssertionError` on the first
                failing check instead of returning ``False``.

        Returns:
            bool: ``True`` if all checks pass, ``False`` otherwise (unless
            ``throw`` is ``True``, in which case an exception is raised).
        """
        num_solutions = len(self)
        bitstring_size = instance.size if instance is not None else self.bitstrings.shape[1]

        def check(condition: bool, message: str) -> bool:
            if condition:
                return True
            logger.warning(message)
            if throw:
                raise AssertionError(message)
            return False

        expected_shapes = (
            ("bitstrings", self.bitstrings, (num_solutions, bitstring_size)),
            ("costs", self.costs, (num_solutions,)),
            ("counts", self.counts, (num_solutions,)),
            ("probabilities", self.probabilities, (num_solutions,)),
        )

        valid = True
        for name, tensor, expected_shape in expected_shapes:
            valid &= check(
                tuple(tensor.shape) == expected_shape,
                f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}",
            )

        if not valid:
            return False

        from qubosolver.utils import _costs

        if instance is not None:
            expected_costs = _costs.batched_quadratic_cost(
                self.bitstrings.to(instance.matrix.dtype), instance.matrix
            )

            valid &= check(
                torch.allclose(self.costs, expected_costs.to(self.costs.dtype)),
                f"costs {self.costs.tolist()} does not match x^T Q x "
                f"{expected_costs.tolist()} for the corresponding bitstrings",
            )

        valid &= check(
            bool(torch.all(self.costs[:-1] <= self.costs[1:])),
            f"costs {self.costs.tolist()} is not sorted in non-decreasing order",
        )

        if num_solutions == 0:
            return valid

        num_unique_bitstrings = self.bitstrings.unique(dim=0).shape[0]
        valid &= check(
            num_unique_bitstrings == num_solutions,
            f"bitstrings contains {num_solutions - num_unique_bitstrings} duplicate row(s)",
        )

        valid &= check(
            bool(torch.all((self.bitstrings == 0) | (self.bitstrings == 1))),
            f"bitstrings {self.bitstrings.tolist()} contains entries other than 0 or 1",
        )

        valid &= check(
            bool(torch.all(self.counts == self.counts.round())),
            f"counts {self.counts.tolist()} contains non-integer values",
        )
        valid &= check(
            bool(torch.all(self.counts > 0)),
            f"counts {self.counts.tolist()} contains non-positive entries",
        )

        expected_probabilities = self.counts / self.counts.sum()
        valid &= check(
            torch.allclose(self.probabilities, expected_probabilities.to(self.probabilities.dtype)),
            f"probabilities {self.probabilities.tolist()} does not match counts "
            f"{self.counts.tolist()} normalised by their sum",
        )

        return valid
