# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Bit-flip preprocessing transforms for QUBO instances with negative interactions.

Quantum (Rydberg) solvers cannot encode attractive interactions, so a QUBO must
have non-negative off-diagonal coefficients to be embeddable.  A change of
variable ``x_i -> 1 - y_i`` on a subset of variables flips the sign of the
interactions incident to it; choosing the subset that removes as much negative
weight as possible is itself a QUBO, ``f^T Q_f f`` with ``Q_f = diag(C 1) - C``,
solved here by local search or tabu search.

[`apply`][] finds the bit flips (by default with a local search on ``Q_f``) and
applies them to the matrix, returning a wrapper [`Instance`][] that
records the flip vector so the solution can later be mapped back with
[`lift`][].  When bit flips
cannot remove *every* negative off-diagonal coefficient, the remaining ones can
be dropped with [`transforms.zeroing.apply`][].

Typical usage:

```python
from qubosolver.transforms import negative_bitflip
from qubosolver.solving import brute_force

flipped_instance = negative_bitflip.apply(instance, time_limit_s=60.0)
flipped_solution = brute_force.solve(flipped_instance)
solution = negative_bitflip.lift(flipped_solution, flipped_instance)
```
"""

from __future__ import annotations

import copy
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch

import qubosolver
from qubosolver import solving
from qubosolver._checks import TYPE_CHECKING
from qubosolver._io import utils as io_utils
from qubosolver.types import (
    Bitstring,
    Bitstrings,
    Matrix,
    Solution,
    bitstring,
    vector,
)

if TYPE_CHECKING:
    from qubosolver._io.utils import Stream


@dataclass(frozen=True)
class _Edge:
    """Non-zero upper-triangular QUBO interaction."""

    i: int
    j: int
    q: float


def _has_negative_offdiagonal(Q: Matrix, eps: float = 0.0) -> bool:
    """Return True if Q has at least one negative off-diagonal coefficient."""
    n = Q.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=Q.device)
    return bool(torch.any(Q[offdiag_mask] < -eps).item())


def _extract_upper_edges(Q: Matrix, eps: float = 0.0) -> list[_Edge]:
    """Extract non-zero upper-triangular off-diagonal QUBO coefficients."""
    n = Q.shape[0]
    edges: list[_Edge] = []

    for i in range(n):
        for j in range(i + 1, n):
            q = float(Q[i, j].item())
            if abs(q) > eps:
                edges.append(_Edge(i=i, j=j, q=q))

    return edges


def _apply_bitflips(
    bitstrings: Bitstrings,
    flips: Bitstring,
) -> Bitstrings:
    """Apply or undo bit flips on bitstrings.

    The operation is its own inverse.
    """
    if bitstrings.numel() == 0:
        return bitstrings
    return bitstrings ^ flips.unsqueeze(0)


def _transform_qubo_with_bitflips(
    Q: Matrix,
    flips: Bitstring,
) -> tuple[Matrix, float]:
    """Transform QUBO coefficients after variable bit flips.

    Convention:
        x_i = y_i       if flips_i = 0
        x_i = 1 - y_i   if flips_i = 1

    The returned offset satisfies:
        x^T Q x = y^T Q_flipped y + offset
    """
    f = flips.to(dtype=Q.dtype)
    s = 1.0 - 2.0 * f
    linear = 2.0 * s * (Q @ f)
    Q_flipped = Q * torch.outer(s, s) + torch.diag(linear)

    offset = float(f @ Q @ f)

    return Q_flipped, offset


def _coefficient_after_bitflip(q: float, fi: int, fj: int) -> float:
    """Return the off-diagonal coefficient sign after bit flips."""
    xor = fi ^ fj
    return (1.0 - 2.0 * xor) * q


def _compute_negative_weight_metrics(
    Q: Matrix,
    flips: Bitstring,
    eps: float = 0.0,
) -> dict[str, Any]:
    """Compute negative off-diagonal count and weight before and after bit flips."""
    edges = _extract_upper_edges(Q, eps=eps)

    neg_count_before = 0
    neg_count_after = 0
    neg_weight_before = 0.0
    neg_weight_after = 0.0

    for edge in edges:
        q = edge.q

        if q < -eps:
            neg_count_before += 1
            neg_weight_before += abs(q)

        q_after = _coefficient_after_bitflip(q, int(flips[edge.i]), int(flips[edge.j]))
        if q_after < -eps:
            neg_count_after += 1
            neg_weight_after += abs(q_after)

    def reduction_pct(before: float, after: float) -> float:
        if before <= 0:
            return 0.0
        return 100.0 * (before - after) / before

    return {
        "n_edges": len(edges),
        "neg_count_before": neg_count_before,
        "neg_count_after": neg_count_after,
        "neg_count_reduction_pct": reduction_pct(float(neg_count_before), float(neg_count_after)),
        "neg_weight_before": neg_weight_before,
        "neg_weight_after": neg_weight_after,
        "neg_weight_reduction_pct": reduction_pct(neg_weight_before, neg_weight_after),
    }


def _bitflip_qubo(instance: qubosolver.Instance, *, eps: float = 0.0) -> qubosolver.Instance:
    """Build the QUBO ``Q_f`` whose minimum gives the best bit flips for `Q`.

    The bit-flip problem picks flips ``f`` minimizing the negative off-diagonal
    weight left after ``x_i -> 1 - x_i``, i.e. ``sum_{i<j} c_ij (f_i XOR f_j)``.
    Since ``f_i XOR f_j = f_i + f_j - 2 f_i f_j``, this is itself the QUBO
    ``f^T Q_f f`` with ``Q_f = diag(C 1) - C``.

    Args:
        instance: QUBO instance; only the upper triangle of its matrix is read.
        eps: Off-diagonal coefficients with ``abs(q) <= eps`` are ignored.

    Returns:
        An instance holding ``Q_f``, in the dtype of `instance`. Its matrix is
        all zeros when no off-diagonal coefficient survives `eps`.
    """
    C = torch.triu(instance.matrix, diagonal=1)
    C = torch.where(C.abs() > eps, C, torch.zeros_like(C))
    C = C + C.T

    return qubosolver.Instance(torch.diag(C.sum(dim=1)) - C)


def _solve_bitflip(
    instance: qubosolver.Instance,
    *,
    method: Literal["local_search", "tabu_search"],
    time_limit: float,
    eps: float,
    rng: torch.Generator | None,
) -> Solution:

    if method not in ("local_search", "tabu_search"):
        msg = f"Unknown bit-flip method {method!r}."
        raise ValueError(msg)

    # time_limit is a budget for local search and tabu search together.
    deadline = time.monotonic() + time_limit
    bitflip_instance = _bitflip_qubo(instance, eps=eps)

    solution = solving.trivial_solution_search.solve(bitflip_instance)
    if solution:
        return solution

    n_starts = 100
    starts = solving.random_sampling.solve(bitflip_instance, max_bitstrings=n_starts, rng=rng)
    # Add the zero bitstring (no flip) as the last start, so it doesn't win
    # ties in the local search; the search doesn't need sorted starts.
    starts.bitstrings[-1].fill_(0)
    starts.costs[-1] = 0.0
    solution = solving.iterative_bitflip_local_search.solve(
        bitflip_instance,
        starts=starts,
        strategy="best_improvement",
        time_limit=time_limit,
    )
    if method != "tabu_search":
        return solution

    return solving.tabu_search.solve(
        bitflip_instance,
        starts=solution.bitstrings,
        max_iter=1000,
        max_no_improve=100,
        time_limit=max(0.0, deadline - time.monotonic()),
    )


class Instance(qubosolver.Instance):
    """A QUBO instance carrying bit-flip preprocessing history.

    Wraps a parent [`qubosolver.Instance`][] whose off-diagonal coefficients may
    contain negative interactions.  Applying [`apply`][] solves the bit-flip problem,
    stores the flip vector here, and exposes the transformed matrix so it can be
    embedded and solved.  [`lift`][] uses the stored state to map a solution
    back onto the original variables.
    """

    def __init__(self, parent_instance: qubosolver.Instance) -> None:
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original QUBO instance (before bit flips).
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(parent_instance.matrix.detach().clone())
        self._parent_instance = copy.deepcopy(parent_instance)

        self.flips: Bitstring = bitstring.zeros(parent_instance.size)
        """Flip vector applied to the parent matrix, all-zero until [`apply`][] populates it."""

        self._metrics: dict[str, Any] = {}
        """Negative off-diagonal count and weight before/after the flips, as computed
        by [`apply`][]."""

        self._offset: float = 0.0
        """Constant term relating the flipped and original QUBO costs,
        $x^T Q x = y^T Q_{flipped} y + offset$."""

    def _write_body(self, f: Stream[bytes]) -> None:
        """Write the matrix, flip vector, solve metadata, and parent instance to `f`."""
        super()._write_body(f)

        buffer = io.BytesIO()
        torch.save(self.flips, buffer)
        io_utils.save_sized_buffer(f, buffer.getbuffer())

        state_json = json.dumps({"offset": self._offset, "metrics": self._metrics})
        io_utils.save_string(f, state_json)

        self._parent_instance.save(f)

    @classmethod
    def _read_body(cls, f: Stream[bytes]) -> Instance:
        """Read back a negative-bitflip instance written by [`_write_body`][]."""
        instance = Instance(qubosolver.Instance._read_body(f))

        buffer = io.BytesIO(io_utils.load_sized_buffer(f))
        instance.flips = torch.load(buffer, weights_only=True)

        state = json.loads(io_utils.load_string(f))
        instance._offset = state["offset"]
        instance._metrics = state["metrics"]

        instance._parent_instance = qubosolver.Instance.load(f)
        return instance


def apply(
    instance: qubosolver.Instance,
    *,
    method: Literal["local_search", "tabu_search"] = "local_search",
    time_limit_s: float = 60.0,
    eps: float = 0.0,
    rng: torch.Generator | None = None,
) -> Instance:
    """Find the bit flips minimizing negative weight and apply them to the QUBO matrix.

    Wraps `instance` in a bit-flip [`Instance`][], finds the flips with `method`,
    and replaces the matrix with its flipped counterpart.  When
    `instance` has no negative off-diagonal coefficient, the wrapper is returned
    unchanged (``flips`` stays all-zero).

    Args:
        instance: The QUBO instance to preprocess.
        method: ``"local_search"``
            ([`solving.iterative_bitflip_local_search.solve`][qubosolver.solving.classical.iterative_bitflip_local_search.solve]
            on ``Q_f``, from random starts plus the no-flip start),
            ``"tabu_search"`` (the same local search, refined with
            [`solving.tabu_search.solve`][qubosolver.solving.classical.tabu_search.solve]).
        time_limit_s: Time limit in seconds. For ``"tabu_search"``, it is shared
            by the local search and the tabu search.
        eps: Tolerance below which a coefficient is treated as zero.
        rng: PyTorch random number generator controlling the random starts of
            the local search.

    Returns:
        A bit-flip instance carrying the flip vector and metrics.

    Raises:
        ValueError: If `method` is unknown.
    """
    if not _has_negative_offdiagonal(instance.matrix, eps=eps):
        return Instance(instance)

    bitflip_solution = _solve_bitflip(
        instance, method=method, time_limit=time_limit_s, eps=eps, rng=rng
    )

    flips = bitflip_solution[0].bitstring
    objective_value = bitflip_solution[0].cost
    metrics = _compute_negative_weight_metrics(instance.matrix, flips, eps)

    Q_flipped, offset = _transform_qubo_with_bitflips(instance.matrix, flips)

    flipped_instance = Instance(instance)
    flipped_instance._matrix = Q_flipped
    flipped_instance.flips = flips
    flipped_instance._metrics = metrics
    flipped_instance._metrics["objective_value"] = objective_value
    flipped_instance._offset = offset

    return flipped_instance


def lift(flipped_solution: Solution, flipped_instance: Instance) -> Solution:
    """Map a solution of the bit-flipped QUBO instance back onto the original variables.

    Undoes the flips recorded on `flipped_instance` (``y_i -> x_i``) and recomputes
    costs against the original (unflipped) matrix.  When no bit flip was applied,
    returns a deep copy of `flipped_solution` unchanged.

    Args:
        flipped_solution: Solution obtained on the bit-flipped instance.
        flipped_instance: The bit-flipped instance produced by [`apply`][].

    Returns:
        A new solution over the original variables.
    """
    flipped = torch.any(flipped_instance.flips != 0)
    if not flipped:
        return copy.deepcopy(flipped_solution)

    solution = Solution()
    solution.bitstrings = _apply_bitflips(flipped_solution.bitstrings, flipped_instance.flips)
    solution.costs = vector.tensor(
        [flipped_instance._parent_instance.cost(b) for b in solution.bitstrings]
    )
    solution.counts = flipped_solution.counts
    solution.probabilities = flipped_solution.probabilities

    return solution
