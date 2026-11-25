# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Exact brute-force QUBO solver by exhaustive enumeration.

Enumerates the ``2^n`` binary assignments of an ``n``-variable QUBO, evaluates
$x^T Q x$ for each, and returns the ``max_bitstrings`` lowest-cost ones.  This
is exact but scales exponentially in ``n``; it is meant for small instances,
validation, and as a ground-truth reference for other solvers.

Enumeration proceeds in fixed-size batches so peak memory stays bounded and a
wall-clock ``time_limit`` can interrupt it between batches.  When the limit is
reached the best assignments found so far are returned, so the result is a valid
(possibly non-optimal) solution rather than an error.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from qubosolver.types import Instance, Solution
from qubosolver.utils import _costs

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Each batch evaluates 2^_BATCH_BITS assignments. Caps peak memory at roughly
# 2^_BATCH_BITS * n floats for the bit matrix regardless of the total 2^n space.
_BATCH_BITS = 16

# Above this instance size, an unbounded time_limit risks an effectively
# unbounded run, since the search space grows as 2^n.
_LARGE_INSTANCE_SIZE = 20


def _decode_bits(indices: NDArray[np.int64], n: int) -> NDArray[np.int8]:
    """Decode integer indices into their ``n``-bit binary representations.

    Bit ``j`` of variable position ``j`` is taken from the corresponding bit of
    the index, most-significant bit first, so index ``0`` maps to all-zeros.

    Args:
        indices: 1-D integer array of assignment indices in ``[0, 2^n)``.
        n: Number of variables (bit width).

    Returns:
        A ``(len(indices), n)`` bitstrings array.
    """
    shifts = np.arange(n - 1, -1, -1)
    return np.asarray((indices[:, np.newaxis] >> shifts) & 1, dtype=np.int8)


def _stable_smallest(costs: NDArray[np.float64], k: int) -> np.ndarray:
    """Indices of the ``k`` smallest `costs`, sorted, ties broken by lower index.

    Same result as ``np.argsort(costs, kind="stable")[:k]``, but only sorts the
    candidates at or below the ``k``-th smallest value (found with
    ``argpartition``), instead of the whole batch.
    """
    if k >= costs.shape[0]:
        return np.argsort(costs, kind="stable")
    kth = costs[np.argpartition(costs, k - 1)[k - 1]]
    candidates = np.flatnonzero(costs <= kth)  # ascending, so the stable sort keeps index order
    return candidates[np.argsort(costs[candidates], kind="stable")[:k]]


def solve(
    instance: Instance,
    *,
    max_bitstrings: int = 1,
    time_limit: float = float("inf"),
) -> Solution:
    """Solve a QUBO exactly by enumerating all ``2^n`` binary assignments.

    Evaluates every assignment's cost and returns the ``max_bitstrings``
    lowest-cost ones, sorted by ascending cost.  Enumeration is batched; when
    ``time_limit`` elapses, the best assignments found so far are returned.

    Args:
        instance: The QUBO instance to solve.
        max_bitstrings: Number of lowest-cost bitstrings to return.  If
            ``max_bitstrings > 2^n``, only ``2^n`` bitstrings are returned.
        time_limit: Wall-clock budget in seconds.  Enumeration stops between
            batches once the budget is exhausted and returns the best solutions
            found so far.  Use ``float("inf")`` for no limit.

    Returns:
        A solution with up to ``max_bitstrings`` bitstrings, their QUBO costs,
            and probabilities, sorted by ascending cost.
    """
    n: int = instance.size
    if n == 0:
        return Solution.zeros(0)

    if n > _LARGE_INSTANCE_SIZE and math.isinf(time_limit):
        logger.warning(
            "brute_force: no time_limit set for a %s-variable instance; "
            "exhaustively enumerating %s assignments with no time limit "
            "may run for an impractically long time. Consider passing a finite "
            "time_limit.",
            n,
            format(1 << n, ","),
        )

    Q = instance.matrix
    total = 1 << n
    deadline = time.perf_counter() + time_limit

    if max_bitstrings == -1:
        max_bitstrings = total

    best_bits = np.zeros((0, n), dtype=Q.dtype)
    best_costs = np.zeros(0)

    # Batch b covers indices [b * 2^low, (b + 1) * 2^low): the low `low` bits run through
    # the same table in every batch and the high `n - low` bits are constant (equal to b).
    # So decode the low bits once into a reused buffer and only rewrite the high bits.
    low = min(n, _BATCH_BITS)
    bits = np.empty((1 << low, n), dtype=Q.dtype)
    bits[:, n - low :] = _decode_bits(np.arange(1 << low, dtype=np.int64), low)

    for batch in range(total >> low):
        bits[:, : n - low] = _decode_bits(np.asarray([batch], dtype=np.int64), n - low)
        costs = _costs.batched_quadratic_cost(bits, Q)

        merged_costs = np.concatenate((best_costs, costs), axis=0)

        # Keep only the current lowest-cost `max_bitstrings` assignments. `top` indexes
        # the virtual concatenation of `best_bits` and `bits`: gather the winners from
        # each side instead of materializing that whole concatenation.
        k = min(max_bitstrings, merged_costs.shape[0])
        top = _stable_smallest(merged_costs, k)
        kept = best_costs.shape[0]
        from_best = top < kept
        new_bits = np.empty((k, n), dtype=Q.dtype)
        new_bits[from_best] = best_bits[top[from_best]]
        new_bits[~from_best] = bits[top[~from_best] - kept]
        best_bits = new_bits
        best_costs = merged_costs[top]

        if time.perf_counter() >= deadline:
            break

    solution = Solution(
        bitstrings=np.asarray(best_bits, dtype=np.int8),
        costs=best_costs,
        counts=np.asarray([1] * best_bits.shape[0], dtype=np.int64),
    )
    return solution._sort_by_cost()._compute_probabilities()
