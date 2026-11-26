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
# 2^_BATCH_BITS * _BATCH_BITS floats for the low-bit table regardless of the total 2^n space.
_BATCH_BITS = 16

# Each loop step evaluates 2^_BLOCK_BITS consecutive batches with one matmul. Larger blocks
# are slower: the (2^_BLOCK_BITS, 2^_BATCH_BITS) intermediates stop fitting in cache.
_BLOCK_BITS = 2

# Above this instance size, an unbounded time_limit risks an effectively
# unbounded run, since the search space grows as 2^n.
_LARGE_INSTANCE_SIZE = 24


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
    # Big-endian bytes put the most-significant bit first; only the `nbytes` low-order
    # bytes are unpacked, so the temporary is (len, n + 7) bits, not (len, 64).
    nbytes = (n + 7) // 8
    as_bytes = indices.astype(">u8").view(np.uint8).reshape(-1, 8)[:, 8 - nbytes :]
    return np.unpackbits(as_bytes, axis=1)[:, nbytes * 8 - n :].astype(np.int8)


def _stable_smallest(costs: NDArray[np.float64], k: int) -> NDArray[np.intp]:
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
    if max_bitstrings == 0:
        return Solution(
            bitstrings=np.zeros((0, n), dtype=np.int8),
            costs=np.zeros(0),
            counts=np.zeros(0, dtype=np.int64),
        )._compute_probabilities()
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

    k = total if max_bitstrings == -1 else min(max_bitstrings, total)
    # Keeping every assignment: no running top-k, just fill one cost array and sort it once.
    keep_all = k == total

    # Batch b covers indices [b * 2^low, (b + 1) * 2^low): the high `m = n - low` bits are
    # constant (h, the binary form of b) and the low bits run through the same table L.
    # With x = [h; l], the cost splits into blocks of Q:
    #   x^T Q x = h^T Q_hh h  +  l^T Q_ll l  +  l^T (Q_hl + Q_lh^T)^T h
    # The l^T Q_ll l term and the (2^low, m) cross matrix P = L (Q_hl + Q_lh^T)^T are the
    # same in every batch, so they are computed once; each batch then needs one scalar
    # and the cross term P @ h.
    low = min(n, _BATCH_BITS)
    m = n - low
    size = 1 << low
    L = _decode_bits(np.arange(size, dtype=np.int64), low).astype(Q.dtype)
    low_costs = _costs.batched_quadratic_cost(L, Q[m:, m:])
    Q_hh = Q[:m, :m]
    P = L @ (Q[:m, m:] + Q[m:, :m].T).T  # (2^low, m): the cross term is P @ h

    # Each step stacks 2^hb consecutive batches: their h are the rows of H, so all their
    # cross terms come from one matmul H @ P^T, which reads P once instead of per batch.
    hb = min(m, _BLOCK_BITS)
    span = size << hb  # assignments per step
    rows = np.arange(span, dtype=np.int64)

    # The running top-k tracks global assignment indices, decoded to bitstrings only at
    # the end. Ties go to the lower index: `best_*` (earlier batches) precede the batch in
    # the merge, and `_stable_smallest` breaks ties by merge position.
    all_costs = np.empty(total if keep_all else 0)
    best_idx = np.zeros(0, dtype=np.int64)
    best_costs = np.zeros(0)
    done = 0

    for block in range(total >> (low + hb)):
        batches = np.arange(block << hb, (block + 1) << hb, dtype=np.int64)
        H = _decode_bits(batches, m).astype(Q.dtype)
        hh = _costs.batched_quadratic_cost(H, Q_hh)
        # (2^hb, 2^low) flattened row by row, so entry r is assignment done + r.
        costs = (low_costs + H @ P.T + hh[:, np.newaxis]).ravel()

        if keep_all:
            all_costs[done : done + span] = costs
        else:
            # Once k are kept, only costs <= the current k-th best can enter (<= keeps ties,
            # still won by the earlier `best_*`); usually few, so the selection is tiny.
            cand = np.flatnonzero(costs <= best_costs[-1]) if best_costs.shape[0] == k else rows
            merged_costs = np.concatenate((best_costs, costs[cand]))
            merged_idx = np.concatenate((best_idx, done + cand))
            top = _stable_smallest(merged_costs, min(k, merged_costs.shape[0]))
            best_idx = merged_idx[top]
            best_costs = merged_costs[top]
        done += span

        if time.perf_counter() >= deadline:
            break

    if keep_all:
        # Evaluated indices are exactly [0, done), so sorted positions are the indices.
        best_idx = np.argsort(all_costs[:done], kind="stable")
        best_costs = all_costs[best_idx]

    solution = Solution(
        bitstrings=_decode_bits(best_idx, n),
        costs=best_costs,
        counts=np.ones(best_idx.shape[0], dtype=np.int64),
    )
    return solution._sort_by_cost()._compute_probabilities()
