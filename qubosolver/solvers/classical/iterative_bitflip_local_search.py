"""Bit-flip local search for QUBO solutions.

This module provides greedy single-bit-flip local search strategies that
improve an existing :class:`~qubosolver.types.Solution` by iteratively
flipping bits, stopping when no flip improves the objective, a maximum
number of iterations is reached, or a shared time budget is exhausted.

The main public entry point is `iterative_bitflip_local_search`, which
applies the selected strategy independently to every bitstring in a solution
and is used as the post-processing step in
:class:`~qubosolver.solvers.BaseSolver`.
"""

from __future__ import annotations

import itertools
from typing import Literal, Iterable
import time
import torch
from collections.abc import Callable
from copy import deepcopy


from qubosolver import Instance, Solution, vector, vectori, Bitstring


def _iterations(n: int) -> Iterable[int]:
    return itertools.count() if n < 0 else range(n)


def _best_improvement_search(
    qubo_func: Callable[[Bitstring], float],
    s: Bitstring,
    rng: torch.Generator | None = None,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> tuple[Bitstring, float]:
    """Improve a single bitstring via best-improvement bit-flip search.

    At each iteration, evaluates all *n* single-bit flips and applies the one
    with the lowest resulting objective value.  Repeats until no flip
    improves the cost (a local minimum is reached), ``max_iterations`` is
    reached, or ``time_limit`` has elapsed.

    Args:
        qubo_func: Callable that maps a :class:`~qubosolver.types.Bitstring`
            to a scalar cost (lower is better).
        s: Binary tensor of shape ``(n,)`` representing the starting solution.
            The tensor is cloned internally; the original is not modified.
        rng: Optional :class:`torch.Generator` used to randomise the order in
            which bit positions are evaluated at each iteration.  When
            provided, positions are visited in a random permutation, which
            can help escape ties and improve diversity.  When ``None``,
            positions are visited in index order ``0, 1, …, n-1``.
        max_iterations: Maximum number of accepted flips. Defaults to no limit.
        time_limit: Maximum time in seconds the search may run, checked once
            per iteration. Defaults to no limit.

    Returns:
        A 2-tuple of:

        * **improved bitstring** — a :class:`~qubosolver.types.Bitstring` of
          shape ``(n,)`` at a local minimum of *qubo_func*.
        * **cost** — the scalar objective value at that local minimum.
    """
    s_current = s.detach().clone()
    current_objective = qubo_func(s_current)
    deadline = time.monotonic() + time_limit

    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break
        best_idx = None
        best_obj = current_objective
        n = s_current.numel()
        if rng is None:
            indices = torch.arange(n).tolist()
        else:
            # option to diversify
            indices = torch.randperm(n, generator=rng).tolist()
        # Evaluate all possible flips, keep best
        for i in indices:
            s_new = s_current.detach().clone()
            s_new[i] = 1 - s_new[i]
            new_objective = qubo_func(s_new)
            if new_objective < best_obj:
                best_obj = new_objective
                best_idx = i
        # If no improvements, stop
        if best_idx is None:
            break
        # Apply best flip
        s_current[best_idx] = 1 - s_current[best_idx]
        current_objective = best_obj
    return s_current, current_objective


def _first_improvement_search(
    qubo_func: Callable[[Bitstring], float],
    s: Bitstring,
    rng: torch.Generator | None = None,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> tuple[Bitstring, float]:
    """Improve a single bitstring via first-improvement bit-flip search.

    At each iteration, applies the first single-bit flip found that reduces
    the objective, instead of scanning all flips for the best one. Repeats
    until no flip improves the cost, ``max_iterations`` is reached, or
    ``time_limit`` has elapsed.

    Args:
        qubo_func: See `_best_improvement_search`.
        s: See `_best_improvement_search`.
        rng: See `_best_improvement_search`.
        max_iterations: See `_best_improvement_search`.
        time_limit: See `_best_improvement_search`.

    Returns:
        A 2-tuple of the improved bitstring and its cost.
    """
    s_current = s.detach().clone()
    current_objective = qubo_func(s_current)
    deadline = time.monotonic() + time_limit

    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break
        improved = False
        n = s_current.numel()
        indices = (
            torch.arange(n).tolist() if rng is None else torch.randperm(n, generator=rng).tolist()
        )

        for i in indices:
            s_new = s_current.detach().clone()
            s_new[i] = 1 - s_new[i]
            new_objective = qubo_func(s_new)

            if new_objective < current_objective:
                s_current = s_new
                current_objective = new_objective
                improved = True
                break

        if not improved:
            break

    return s_current, current_objective


def _greedy_sweep_search(
    qubo_func: Callable[[Bitstring], float],
    s: Bitstring,
    rng: torch.Generator | None = None,
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> tuple[Bitstring, float]:
    """Improve a single bitstring via greedy-sweep bit-flip search.

    At each iteration, applies every single-bit flip found to improve the
    objective (relative to the state at the start of that sweep), instead of
    stopping at the first or the single best one. Repeats until a sweep makes
    no improvement, ``max_iterations`` is reached, or ``time_limit`` has
    elapsed.

    Args:
        qubo_func: See `_best_improvement_search`.
        s: See `_best_improvement_search`.
        rng: See `_best_improvement_search`.
        max_iterations: See `_best_improvement_search`.
        time_limit: See `_best_improvement_search`.

    Returns:
        A 2-tuple of the improved bitstring and its cost.
    """
    s_current = s.detach().clone()
    current_objective = qubo_func(s_current)
    deadline = time.monotonic() + time_limit

    for _ in _iterations(max_iterations):
        if time.monotonic() > deadline:
            break
        improved = False
        n = s_current.numel()
        indices = (
            torch.arange(n).tolist() if rng is None else torch.randperm(n, generator=rng).tolist()
        )

        for i in indices:
            s_new = s_current.detach().clone()
            s_new[i] = 1 - s_new[i]
            new_objective = qubo_func(s_new)

            if new_objective < current_objective:
                s_current = s_new
                current_objective = new_objective
                improved = True

        if not improved:
            break

    return s_current, current_objective


_STRATEGIES: dict[str, Callable[..., tuple[Bitstring, float]]] = {
    "best_improvement": _best_improvement_search,
    "first_improvement": _first_improvement_search,
    "greedy_sweep": _greedy_sweep_search,
}


def solve(
    instance: Instance,
    solution: Solution,
    *,
    strategy: Literal["greedy_sweep", "best_improvement", "first_improvement"] = "greedy_sweep",
    max_iterations: int = -1,
    time_limit: float = float("inf"),
) -> Solution:
    """Improve every bitstring in `solution` via single-bit-flip local search.

    After refinement, duplicate bitstrings that were driven to
    the same local minimum are merged: their counts are summed, the minimum
    cost is retained, and sampling probabilities are recomputed from the merged
    counts.

    ``time_limit`` is a *global* budget shared by the whole batch of
    bitstrings in *solution*, not a per-bitstring limit: a `deadline` is
    computed once before iterating over the batch, and every bitstring's
    search shares it. Once the deadline passes, remaining bitstrings in the
    batch are returned unchanged (with their original cost) instead of being
    searched.

    Args:
        instance: The instance used to evaluate bitstring costs.
        solution: The solution to refine.
        strategy: Which local-search strategy to use: ``"best_improvement"``,
            ``"first_improvement"``, or ``"greedy_sweep"``. Defaults to
            ``"greedy_sweep"``.
        max_iterations: Maximum number of accepted flips per bitstring.
            Defaults to no limit.
        time_limit: Maximum total time in seconds for the whole batch.
            Defaults to no limit.

    Returns:
        A new solution with updated `bitstrings`, `costs`, `counts`, and `probabilities` reflecting the locally optimal results.

    Raises:
        ValueError: If `strategy` is not one of the supported strategies.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown postprocessing strategy: {strategy}")

    solution = deepcopy(solution)

    # If there are no bitstrings, return the solution unchanged.
    if not solution:
        return solution

    # Define an objective function that uses the existing cost method.
    def qubo_objective(s_arr: Bitstring) -> float:
        # Convert the solution array to a list of integers
        return instance.cost(s_arr)

    if solution.costs.numel() == 0:
        solution._compute_costs(instance.matrix)._sort_by_cost()

    search_fn = _STRATEGIES[strategy]
    deadline = time.monotonic() + time_limit

    for i, sol in enumerate(solution):
        # Get the current solution (row) as a numpy array of integers.
        s_orig = sol.bitstring

        time_limit = deadline - time.monotonic()
        if time_limit < 0.0:
            break

        # Apply bit-flip local search to improve the solution.
        solution.bitstrings[i, :], solution.costs[i] = search_fn(
            qubo_objective, s_orig, rng=None, max_iterations=max_iterations, time_limit=time_limit
        )

    solution.deduplicate()

    return solution
