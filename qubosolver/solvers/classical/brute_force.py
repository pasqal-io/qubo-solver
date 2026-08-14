"""Exact brute-force QUBO solver by exhaustive enumeration.

Enumerates the ``2^n`` binary assignments of an ``n``-variable QUBO, evaluates
``x^T Q x`` for each, and returns the ``max_bitstrings`` lowest-cost ones.  This
is exact but scales exponentially in ``n``; it is meant for small instances,
validation, and as a ground-truth reference for other solvers.

Enumeration proceeds in fixed-size batches so peak memory stays bounded and a
wall-clock ``time_limit`` can interrupt it between batches.  When the limit is
reached the best assignments found so far are returned, so the result is a valid
(possibly non-optimal) solution rather than an error.

This solver is intentionally **not** wired into the object-oriented
[`Solver`][qubosolver.solvers.Solver] dispatcher; call `brute_force`
directly.
"""

from __future__ import annotations

import logging
import math
import time

import torch

from qubosolver.types import Bitstrings, Instance, Solution, Vectori, bitstrings, vector, vectori
from qubosolver.utils import _costs

logger = logging.getLogger(__name__)

# Number of assignments evaluated per batch. Caps peak memory at roughly
# _BATCH_SIZE * n bytes for the bit matrix regardless of the total 2^n space.
_BATCH_SIZE = 1 << 16

# Above this instance size, an unbounded time_limit risks an effectively
# unbounded run, since the search space grows as 2^n.
_LARGE_INSTANCE_SIZE = 20


def _decode_bits(indices: Vectori, n: int) -> Bitstrings:
    """Decode integer indices into their ``n``-bit binary representations.

    Bit ``j`` of variable position ``j`` is taken from the corresponding bit of
    the index, most-significant bit first, so index ``0`` maps to all-zeros.

    Args:
        indices: 1-D integer tensor of assignment indices in ``[0, 2^n)``.
        n: Number of variables (bit width).

    Returns:
        A ``(len(indices), n)`` bitstrings tensor.
    """
    shifts = torch.arange(n - 1, -1, -1, device=indices.device)
    return bitstrings.from_torch((indices.unsqueeze(1) >> shifts) & 1)


def brute_force(
    instance: Instance,
    *,
    max_bitstrings: int = 1,
    time_limit: float = 60.0,
) -> Solution:
    """Solve a QUBO exactly by enumerating all ``2^n`` binary assignments.

    Evaluates every assignment's cost and returns the ``max_bitstrings``
    lowest-cost ones, sorted by ascending cost.  Enumeration is batched; when
    ``time_limit`` elapses, the best assignments found so far are returned.

    Args:
        instance: The QUBO instance to solve.
        max_bitstrings: Number of lowest-cost bitstrings to return.  The result
            may contain fewer when ``2^n < max_bitstrings``.
        time_limit: Wall-clock budget in seconds.  Enumeration stops between
            batches once the budget is exhausted and returns the best solutions
            found so far.  Use ``float("inf")`` for no limit.

    Returns:
        A solution with up to ``max_bitstrings`` bitstrings, their QUBO costs,
        and probabilities, sorted by ascending cost.  Empty for a zero-variable
        instance.
    """
    n: int = instance.size
    if n == 0:
        return Solution()

    if n > _LARGE_INSTANCE_SIZE and math.isinf(time_limit):
        logger.warning(
            f"brute_force: no time_limit set for a {n}-variable instance; "
            f"exhaustively enumerating {1 << n:,} assignments with no time limit "
            "may run for an impractically long time. Consider passing a finite "
            "time_limit."
        )

    Q = instance.matrix
    total = 1 << n
    deadline = time.perf_counter() + time_limit

    if max_bitstrings == -1:
        max_bitstrings = total

    best_bits = bitstrings.zeros(0, n, device=Q.device).to(Q.dtype)
    best_costs = vector.zeros(0)

    for start in range(0, total, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, total)
        indices = vectori.from_torch(torch.arange(start, stop, device=Q.device))
        bits = _decode_bits(indices, n).to(Q.dtype)
        costs = _costs.batched_quadratic_cost(bits, Q)

        merged_bits = torch.cat((best_bits, bits), dim=0)
        merged_costs = torch.cat((best_costs, costs), dim=0)

        # Keep only the current lowest-cost `max_bitstrings` assignments.
        k = min(max_bitstrings, merged_costs.shape[0])
        top = torch.topk(merged_costs, k, largest=False).indices
        best_bits = merged_bits[top]
        best_costs = merged_costs[top]

        if time.perf_counter() >= deadline:
            break

    solution = Solution(
        bitstrings=bitstrings.from_torch(best_bits),
        costs=best_costs,
        counts=vectori.tensor([1] * best_bits.shape[0]),
    )
    return solution._sort_by_cost()._compute_probabilities()
