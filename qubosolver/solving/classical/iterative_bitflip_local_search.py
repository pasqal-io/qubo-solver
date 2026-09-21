"""Bit-flip local search for QUBO solutions.

This module provides greedy single-bit-flip local search strategies that
improve a batch of candidate bitstrings by iteratively flipping bits,
stopping when no flip improves the objective, a maximum number of
iterations is reached, or a shared time budget is exhausted.

The main public entry point is [`solve`][], which applies the selected
strategy to every bitstring in an existing [`Solution`][], or to a batch of
uniformly random candidates generated on the fly. It is used as a
post-processing step in [`Solver`][]. ``"best_improvement"`` runs every
bitstring in lockstep as a single batched search; ``"first_improvement"``
and ``"greedy_sweep"`` improve each bitstring independently.
"""

from __future__ import annotations

import itertools
from typing import Literal, Iterable
import time
import numpy as np
import torch
from collections.abc import Callable
from copy import deepcopy


from qubosolver import Instance, Solution, Bitstring, Matrix, bitstrings, Bitstrings, bitstring
from qubosolver.utils._costs import _flip_deltas
from .random_sampling import solve as random_sampling_solve


def _iterations(n: int) -> Iterable[int]:
    return itertools.count() if n < 0 else range(n)


# Deliberately not implemented: applying several flips per round.
#
# `_best_improvement_search_batch` spends ~86% of a round in `_flip_deltas`,
# recomputing the full `(m, n)` delta table to justify a single flip per row.
# Those deltas do not have to be recomputed from scratch: since `x^T Q x` is
# quadratic, its expansion terminates at second order, so flipping bit `j`
# shifts bit `i`'s delta by exactly
#
#     dE_i += 2 * s_i * s_j * Q[i, j]        with s_k = 1 - 2 * x_k
#
# i.e. one row of the Hessian (`Q` *is* the Hessian here, up to a factor 2).
# The term is exactly additive over flips for symmetric `Q` -- there is no
# residual cross-term -- so `dE` can be carried forward with a rank-1 update
# (one `Q[j, :]` gather, measured ~26x cheaper than a fresh `_flip_deltas`)
# and several flips applied per round, each re-checked against the corrected
# deltas and taken only while strictly negative. That preserves the descent
# guarantee and local-minimality exactly.
#
# A working version measured ~2.4x on top of row compaction (rounds 2541 -> 81
# at n=4000, m=200, bit-identical costs). It was dropped on purpose: it is
# second-order machinery on a deliberately naive local search. Revisit only
# if this function becomes a real bottleneck.


def _best_improvement_search_batch(
    Q: Matrix,
    X: Bitstrings,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> Bitstrings:
    """Improve every row of `X` via best-improvement bit-flip search, in lockstep.

    At each round, every row that still has an improving flip applies its own
    best-improving flip in a single vectorized step; rows already at a local
    optimum are left untouched. The whole batch stops together once *no* row
    has an improving flip left, `max_iterations` rounds have been run, or
    `time_limit` has elapsed.

    Args:
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.
        X: Binary tensor of shape ``(m, n)``, one starting solution per row.
            The tensor is cloned internally; the original is not modified.
        max_iterations: Maximum number of rounds. Each round applies one flip
            to every row that still has an improving one, so a row may end
            up with fewer flips than `max_iterations` if it converges first,
            and different rows can converge after different numbers of
            flips even under the same round cap. Defaults to no limit.
        time_limit: Maximum time in seconds the search may run, checked once
            per round. Defaults to no limit.

    Returns:
        A `Bitstring` of shape ``(m, n)``, each row at a local minimum of
            `Q`'s quadratic form (or as close as `max_iterations`/
            `time_limit` allowed).
    """
    X = X.detach().clone().to(Q)
    QX = X @ Q
    rows = torch.arange(X.shape[0], device=X.device)
    deadline = time.monotonic() + time_limit
    diagonal = Q.diagonal()
    dE_buffer = torch.empty_like(QX)

    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break

        dE = _flip_deltas(Q, X, QX, diagonal=diagonal, out=dE_buffer)
        best_dE, best_idx = dE.min(dim=1)

        improving = best_dE < 0.0
        if not bool(improving.any()):
            break

        idx = best_idx[improving]
        xi = X[rows[improving], idx]
        step = 1.0 - 2.0 * xi
        X[rows[improving], idx] = xi + step
        QX[improving] += step.unsqueeze(1) * Q[idx, :]

    return bitstrings.round(X)


def _first_improvement_search(
    Q_torch: Matrix,
    s: Bitstring,
    rng: torch.Generator | None = None,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> Bitstring:
    """Improve a single bitstring via first-improvement bit-flip search.

    At each iteration, applies the first single-bit flip found that reduces
    the objective, instead of scanning all flips for the best one. Repeats
    until no flip improves the cost, `max_iterations` is reached, or
    `time_limit` has elapsed.

    Args:
        Q_torch: Symmetric QUBO coefficient matrix of shape ``(n, n)``.
        s: Binary tensor of shape ``(n,)``, the starting solution. The
            tensor is cloned internally; the original is not modified.
        rng: Optional Generator
            used to randomize the order in which bit positions are evaluated
            at each iteration. When provided, positions are visited in a
            random permutation, which can help escape ties and improve
            diversity. When ``None``, positions are visited in index order
            ``0, 1, …, n-1``.
        max_iterations: Maximum number of accepted flips. Defaults to `-1`,
            i.e. no limit.
        time_limit: Maximum time in seconds the search may run, checked once
            per iteration. Defaults to no limit.

    Returns:
        The improved bitstring, at a local minimum of `Q`'s quadratic form
            (or as close as `max_iterations`/`time_limit` allowed). Its cost
            is not returned: the deltas accumulated in `Qx` across flips can
            drift from the true cost by a few ULPs, so callers should
            recompute it exactly instead of trusting an incremental value.
    """
    Q = Q_torch.detach().cpu().numpy()
    x = s.to(Q_torch).detach().cpu().numpy().copy()
    deadline = time.monotonic() + time_limit
    n = x.shape[0]
    indices = np.arange(n)
    np_rng = None
    if rng is not None:
        seed = int(torch.randint(0, 2**63 - 1, (1,), generator=rng).item())
        np_rng = np.random.default_rng(seed)

    # NumPy gives the cheapest combination here too (see `_greedy_sweep_search`
    # above): scalar element access is far cheaper than torch's per-element
    # dispatch/.item() sync, and the row update stays a vectorized C loop.
    Qx = Q @ x
    diagonal = Q.diagonal()

    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break

        dE = 2.0 * Qx - 4.0 * Qx * x + diagonal

        if np_rng is not None:
            indices = np_rng.permutation(n)
            dE = dE[indices]

        # Pick the first improving bit *in visit order* without a Python-level
        # scan: argmax on the boolean mask returns the lowest index that is
        # True, and 0 when none is (hence the explicit re-check of that entry).
        improving = dE < 0.0
        position = int(np.argmax(improving))
        if not improving[position]:
            break

        i = int(indices[position]) if np_rng is not None else position
        x_i = x[i]
        step = 1.0 - 2.0 * x_i
        x[i] = x_i + step
        Qx += step * Q[i, :]

    return bitstring.round(torch.from_numpy(x).to(device=Q_torch.device, dtype=Q_torch.dtype))


def _greedy_sweep_search(
    Q_torch: Matrix,
    s: Bitstring,
    rng: torch.Generator | None = None,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> Bitstring:
    """Improve a single bitstring via greedy-sweep bit-flip search.

    At each iteration, sweeps through every bit position and applies every
    flip found to improve the objective relative to the current state at the
    time it is evaluated, instead of stopping at the first or the single best
    one. Repeats until a sweep makes no improvement, `max_iterations` is
    reached, or `time_limit` has elapsed.

    Args:
        Q_torch: Symmetric QUBO coefficient matrix of shape ``(n, n)``.
        s: Binary tensor of shape ``(n,)``, the starting solution. The
            tensor is cloned internally; the original is not modified.
        rng: Optional generator
            used to randomize the order in which bit positions are visited
            within each sweep. When provided, positions are visited in a
            random permutation, which can help escape ties and improve
            diversity. When ``None``, positions are visited in index order
            ``0, 1, …, n-1``.
        max_iterations: Maximum number of sweeps. Defaults to `-1`, i.e. no
            limit.
        time_limit: Maximum time in seconds the search may run, checked once
            per sweep. Defaults to no limit.

    Returns:
        The improved bitstring, at a local minimum of `Q`'s quadratic form
            (or as close as `max_iterations`/`time_limit` allowed). Its cost
            is not returned: the deltas accumulated in `Qx` across flips can
            drift from the true cost by a few ULPs, so callers should
            recompute it exactly instead of trusting an incremental value.
    """
    Q = Q_torch.detach().cpu().numpy()
    x = s.to(Q_torch).detach().cpu().numpy().copy()
    deadline = time.monotonic() + time_limit
    n = x.shape[0]
    visit_order = range(n)
    np_rng = None
    if rng is not None:
        seed = int(torch.randint(0, 2**63 - 1, (1,), generator=rng).item())
        np_rng = np.random.default_rng(seed)

    # NumPy gives the cheapest combination for this sweep: scalar element
    # access is far cheaper than torch's per-element dispatch/.item() sync,
    # and row updates (`Qx += step * Q[i, :]`) still run as a vectorized
    # C loop rather than the Python-level loop a plain list would need.
    # Measured ~5-12x faster than both a Python-list and an all-tensor
    # version of this same sweep at n=4000.
    Qx = Q @ x
    diagonal = Q.diagonal()

    # The sweep is inherently sequential (each accepted flip changes the
    # deltas the remaining bits see), so it cannot be vectorized.
    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break

        improved = False

        if np_rng is not None:
            visit_order = np_rng.permutation(n)  # type: ignore[assignment]

        # Flipping bit i changes the deltas seen by the remaining bits in
        # this sweep, so each bit's delta must be recomputed from the
        # incrementally maintained Qx right before it is evaluated, rather
        # than computed once for the whole sweep up front.
        for i in visit_order:
            x_i = x[i]
            step = 1.0 - 2.0 * x_i

            if 2.0 * Qx[i] * step + diagonal[i] < 0.0:
                x[i] = x_i + step
                Qx += step * Q[i, :]
                improved = True

        if not improved:
            break

    return bitstring.round(torch.from_numpy(x).to(device=Q_torch.device, dtype=Q_torch.dtype))


_ROW_STRATEGIES: dict[str, Callable[..., Bitstring]] = {
    "first_improvement": _first_improvement_search,
    "greedy_sweep": _greedy_sweep_search,
}


def solve(
    instance: Instance,
    *,
    starts: Solution | int = 1,
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"] = "greedy_sweep",
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> Solution:
    """Improve every bitstring in `starts` via single-bit-flip local search.

    Bitstrings driven to the same local minimum are merged afterwards via
    [`deduplicate`][qubosolver.Solution.deduplicate].

    `time_limit` is a *global* budget for the whole batch of bitstrings in
    `starts`, not a per-bitstring limit. Once it is exhausted, any
    remaining bitstrings are left unchanged, with their original cost.

    Args:
        instance: The instance used to evaluate bitstring costs.
        starts: Either the [`Solution`][] to refine, or an ``int`` giving
            the number of uniformly random candidate bitstrings to draw
            via [`random_sampling.solve`][],
            which may return fewer than requested after deduplication. This
            random draw is *not* reproducible via a caller-supplied `rng`;
            callers who need reproducibility should sample their own
            [`Solution`][] (e.g. with a seeded [`random_sampling.solve`][])
            and pass it in directly.
        strategy: Which local-search strategy to use: ``"best_improvement"``,
            ``"first_improvement"``, or ``"greedy_sweep"``.
        max_iterations: For ``"first_improvement"`` and ``"greedy_sweep"``,
            the maximum number of accepted flips per bitstring. For
            ``"best_improvement"``, the maximum number of *batch rounds*:
            each round applies one flip to every bitstring that still has
            an improving one, so a bitstring may converge in fewer rounds
            than this cap, and different bitstrings can converge after
            different numbers of flips under the same cap. Defaults to
            `-1`, i.e. no limit.
        time_limit: Maximum total time in seconds for the whole batch.
            Defaults to `float('inf')`, i.e. no limit.

    Returns:
        A new solution with updated `bitstrings`, `costs`, `counts`,
            and `probabilities` reflecting the locally optimal results.

    Raises:
        ValueError: If `strategy` is not one of the supported strategies.
    """
    if strategy != "best_improvement" and strategy not in _ROW_STRATEGIES:
        raise ValueError(f"Unknown postprocessing strategy: {strategy}")

    if isinstance(starts, int):
        solution = random_sampling_solve(instance, max_bitstrings=starts)
    else:
        solution = deepcopy(starts)

    # If there are no bitstrings, return the solution unchanged.
    if not solution:
        return solution

    if strategy == "best_improvement":
        # best_improvement is batched over every row at once instead of
        # looping row by row, so it needs its own dispatch path here.
        solution.bitstrings = _best_improvement_search_batch(
            instance.matrix,
            solution.bitstrings,
            max_iterations=max_iterations,
            time_limit=time_limit,
        )
    else:
        search_fn = _ROW_STRATEGIES[strategy]
        deadline = time.monotonic() + time_limit

        for i, sol in enumerate(solution):
            # Get the current solution (row) as a numpy array of integers.
            s_orig = sol.bitstring

            # `time_limit` is a budget for the whole batch, so each row gets
            # only what is left of it, not a fresh full budget.
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break

            # Apply bit-flip local search to improve the solution.
            solution.bitstrings[i, :] = search_fn(
                instance.matrix,
                s_orig,
                rng=None,
                max_iterations=max_iterations,
                time_limit=remaining,
            )

    # The incremental deltas used above can drift from the true x^T Q x by a
    # few ULPs; recompute exactly rather than trusting the accumulated value.
    solution._compute_costs(instance.matrix)
    solution.deduplicate()

    return solution
