"""Tabu Search solver for QUBO problems.

A single-neighborhood tabu search that
explores bit-flip moves in parallel across multiple starting points.
"""

from __future__ import annotations

import time

import torch

from qubosolver.types import Bitstrings, Instance, Solution, bitstrings, vector, vectori
from qubosolver.utils._costs import _flip_deltas

from .trivial_solution_search import _zero_length_solution

# How often the incremental QX/f_current tracking is refreshed by an exact
# recompute. Bounds the rounding drift accumulated by the incremental update
# without materially adding to the per-iteration cost.
_REFRESH_EVERY = 128


def solve(
    instance: Instance,
    *,
    starts: Bitstrings | int = 1,
    max_iter: int = 100,
    tabu_tenure: int = 7,
    max_no_improve: int = 20,
    time_limit: float = float("inf"),
) -> Solution:
    """Perform Tabu Search on a QUBO instance to find low-cost bitstrings.

    Runs one independent search per row of `starts`, each exploring
    single-bit-flip neighbors from its own starting point.  A tabu list
    prevents revisiting recently flipped bits; aspiration overrides the tabu
    restriction whenever a move yields a new global best.  All independent
    runs share the same stopping criteria and are deduplicated before being
    returned.

    Args:
        instance: The QUBO instance providing the cost matrix.
        starts: Either a batch of initial binary solutions, one row per
            independent run, each of length ``n``, or an ``int`` giving the
            number of uniformly random starts to generate. This random draw
            is *not* reproducible via a caller-supplied `rng`; callers who
            need reproducibility should sample their own [`Bitstrings`][]
            (e.g. with a seeded [`bitstrings.rand`][]) and pass it in
            directly.
        max_iter: Maximum number of search iterations.
        tabu_tenure: Number of iterations a bit-flip move stays tabu.
        max_no_improve: Maximum consecutive iterations without improvement
            before a run is considered stagnated.  Search stops early when
            **all** independent runs have stagnated.
        time_limit: Wall-clock time budget in seconds. Defaults to
            ``float('inf')`` (no limit).

    Returns:
        Deduplicated best bitstrings found across all runs, together with their objective
            values and occurrence counts, sorted by ascending cost.

    Raises:
        ValueError: If `starts` bitstrings do not have length `instance.size`.
    """
    if isinstance(starts, int):
        starts = bitstrings.rand(starts, instance.size)

    if starts.shape[1] != instance.size:
        raise ValueError(
            f"starts has bitstrings of length {starts.shape[1]}, "
            f"but instance.size is {instance.size}."
        )

    if instance.size == 0:
        return _zero_length_solution(starts.shape[0])

    Q = instance.matrix
    device = Q.device
    n_bitstrings, n = starts.shape

    # Repeat x0 for each parallel run
    X = starts.detach().clone().to(Q)
    QX = X @ Q
    f_current = (X * QX).sum(dim=1)
    x_best = X.clone()
    f_best = f_current.clone()

    # Tabu list per run and bit
    tabu_list = torch.zeros((n_bitstrings, n), dtype=torch.int64, device=device)
    iter_since_last_improve = torch.zeros(n_bitstrings, dtype=torch.int64, device=device)

    deadline = time.perf_counter() + time_limit
    rows = torch.arange(n_bitstrings, device=device)
    cols = torch.arange(n, device=device)
    diagonal = Q.diagonal()
    dE_buffer = torch.empty_like(QX)

    for iteration in range(max_iter):
        if time.perf_counter() >= deadline:
            break

        # `f_current` and `QX` are accumulated incrementally below, so each
        # step adds a rounding error that leaves them a few ULPs off the true
        # x^T Q x. Recomputing them exactly every _REFRESH_EVERY iterations
        # keeps that error from growing unbounded, at the cost of one extra
        # matmul amortized over many iterations.
        if iteration % _REFRESH_EVERY == 0:
            QX = X @ Q
            f_current = (X * QX).sum(dim=1)

        # Tabu tenure counts down to 0 every iteration, regardless of whether
        # a move is made; a bit is tabu while its counter is still positive.
        tabu_list.sub_(1).clamp_(min=0)

        # Delta of each candidate one-bit-flip move, for every run at once;
        # avoids recomputing the full x^T Q x per candidate.
        dE = _flip_deltas(Q, X, QX, diagonal=diagonal, out=dE_buffer)
        f_candidates = f_current.unsqueeze(1) + dE

        # Tabu and aspiration
        tabu_mask = tabu_list > 0
        aspiration_mask = f_candidates < f_best.unsqueeze(1)
        allowed = (~tabu_mask) | aspiration_mask

        # Mask out disallowed moves
        f_masked = torch.where(allowed, f_candidates, torch.inf)

        # Pick best move per run
        best_moves = torch.argmin(f_masked, dim=1)
        move_mask = cols.unsqueeze(0) == best_moves.unsqueeze(1)

        # Apply the best move
        xi = X[rows, best_moves]
        step = 1.0 - 2.0 * xi
        X[rows, best_moves] = xi + step
        QX += step.unsqueeze(1) * Q[best_moves, :]
        # Read the new cost from `f_candidates`, not from the tabu-masked
        # `f_masked`: when every move of a run is disallowed, `f_masked` is
        # all-`inf` for that row and its minimum is the `inf` sentinel, not a
        # real cost. `f_current` is persistent state, so assigning that
        # sentinel here would make every later `f_current + dE` `inf` too --
        # including at allowed positions -- which also disables aspiration
        # (`inf < f_best` is never true) and stops the run searching.
        f_current = f_candidates[rows, best_moves]
        tabu_list[move_mask] = tabu_tenure

        # Update best solutions
        improved = f_current < f_best
        x_best[improved] = X[improved]
        f_best[improved] = f_current[improved]
        iter_since_last_improve += 1
        iter_since_last_improve[improved] = 0

        # Early stop if all stagnated
        if torch.all(iter_since_last_improve >= max_no_improve):
            break

    # `f_best` was accumulated incrementally, drifting from the true x^T Q x
    # by up to _REFRESH_EVERY steps of rounding error. `deduplicate` picks the
    # row to keep per bitstring based on that drifted cost, so skip its own
    # recompute (`update=False`) and instead recompute exactly via `_update`
    # right after.
    solution = Solution(
        bitstrings=bitstrings.as_tensor(x_best),
        costs=f_best,
        counts=vectori.zeros(n_bitstrings).fill_(1),
        probabilities=vector.zeros(n_bitstrings).fill_(1.0 / n_bitstrings),
    )
    return solution.deduplicate(update=False)._update(instance)
