"""Tabu Search solver for QUBO problems.

Provides `tabu_search`, a single-neighbourhood tabu search that
explores bit-flip moves in parallel across multiple starting points.
"""

from __future__ import annotations

import time

import torch

from qubosolver.types import Instance, Solution, Bitstrings, bitstrings
from qubosolver.utils import _costs


def solve(
    qubo: Instance,
    start: Bitstrings,
    *,
    max_iter: int = 100,
    tabu_tenure: int = 7,
    max_no_improve: int = 20,
    time_limit: float = float("inf"),
) -> Solution:
    """Perform Tabu Search on a QUBO instance to find low-cost bitstrings.

    Runs one independent search per row of ``start``, each exploring
    single-bit-flip neighbours from its own starting point.  A tabu list
    prevents revisiting recently flipped bits; aspiration overrides the tabu
    restriction whenever a move yields a new global best.  All independent
    runs share the same stopping criteria and are deduplicated before being
    returned.

    Args:
        qubo (Instance): The QUBO instance providing the cost matrix.
        start (Bitstrings): Initial binary solutions, one row per independent
            run, each of length ``n``.
        max_iter (int): Maximum number of search iterations. Defaults to 100.
        tabu_tenure (int): Number of iterations a bit-flip move stays tabu.
            Defaults to 7.
        max_no_improve (int): Maximum consecutive iterations without improvement
            before a run is considered stagnated.  Search stops early when
            **all** independent runs have stagnated. Defaults to 20.
        time_limit (float): Wall-clock time budget in seconds.  Defaults to
            ``float('inf')`` (no limit).

    Returns:
        Deduplicated best bitstrings found across all runs, together with their objective values and occurrence counts.
    """
    Q = qubo.matrix
    device = Q.device
    n_bitstrings, n = start.shape

    # Repeat x0 for each parallel run
    x_current = start.detach().clone()
    f_current = _costs.batched_quadratic_cost(x_current.to(Q), Q)
    x_best = x_current.clone()
    f_best = f_current.clone()

    # Tabu list per run and bit
    tabu_list = torch.zeros((n_bitstrings, n), dtype=torch.int64, device=device)
    iter_since_last_improve = torch.zeros(n_bitstrings, dtype=torch.int64, device=device)

    deadline = time.perf_counter() + time_limit

    for iteration in range(max_iter):
        if time.perf_counter() >= deadline:
            break

        # Generate all neighbor candidates for each bit flip
        flips = torch.eye(n, dtype=torch.int64, device=device).unsqueeze(0)
        x_neighbors = x_current.unsqueeze(1).clone()
        x_neighbors = (x_neighbors + flips) % 2  # each bit flipped
        f_candidates = _costs.batched_quadratic_cost(x_neighbors.view(-1, n).to(Q), Q).view(
            n_bitstrings, n
        )

        # Tabu and aspiration
        tabu_mask = tabu_list > iteration
        aspiration_mask = f_candidates < f_best.unsqueeze(1)
        allowed = (~tabu_mask) | aspiration_mask

        # Mask out disallowed moves
        f_masked = torch.where(allowed, f_candidates, torch.inf)

        # Pick best move per run
        best_costs, best_moves = torch.min(f_masked, dim=1)
        move_mask = torch.arange(n, device=device).unsqueeze(0) == best_moves.unsqueeze(1)

        # Apply the best move
        x_current = (x_current + move_mask.to(torch.int64)) % 2
        f_current = best_costs
        tabu_list = torch.where(move_mask, iteration + tabu_tenure, tabu_list)

        # Update best solutions
        improved = f_current < f_best
        x_best = torch.where(improved.unsqueeze(1), x_current, x_best)
        f_best = torch.where(improved, f_current, f_best)
        iter_since_last_improve = torch.where(improved, 0, iter_since_last_improve + 1)

        # Early stop if all stagnated
        if torch.all(iter_since_last_improve >= max_no_improve):
            break

    # Get unique final solutions
    uniq, counts = torch.unique(x_best, dim=0, return_counts=True)
    costs = _costs.batched_quadratic_cost(uniq.to(Q), Q)

    solution = (
        Solution(
            bitstrings=bitstrings.from_torch(uniq),
            costs=costs,
            counts=counts,
        )
        ._sort_by_cost()
        ._compute_probabilities()
    )

    return solution
