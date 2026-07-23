"""Bit-flip local search for QUBO solutions.

This module provides a greedy single-bit-flip local search that improves an
existing :class:`~qubosolver.types.Solution` by iteratively flipping the
bit that yields the greatest cost reduction, stopping when no single flip
improves the objective.

The main public entry point is `iterative_bitflip_local_search`, which
applies this search independently to every bitstring in a solution and is used
as the post-processing step in :class:`~qubosolver.solvers.BaseSolver`.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from copy import deepcopy


from qubosolver import Instance, Solution, bitstrings, vector, vectori, Bitstring


def _bit_flip_local_search(
    qubo_func: Callable[[Bitstring], float],
    s: Bitstring,
    rng: torch.Generator | None = None,
) -> tuple[Bitstring, float]:
    """Improve a single bitstring via best-improvement bit-flip search.

    At each iteration, evaluates all *n* single-bit flips and applies the one
    with the lowest resulting objective value.  Repeats until no flip reduces
    the cost (a local minimum is reached).

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

    Returns:
        A 2-tuple of:

        * **improved bitstring** — a :class:`~qubosolver.types.Bitstring` of
          shape ``(n,)`` at a local minimum of *qubo_func*.
        * **cost** — the scalar objective value at that local minimum.
    """
    s_current = s.detach().clone()
    current_objective = qubo_func(s_current)
    while True:
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


def iterative_bitflip_local_search(instance: Instance, solution: Solution) -> Solution:
    """Improve every bitstring in `solution` via single-bit-flip local search.

    After refinement, duplicate bitstrings that were driven to
    the same local minimum are merged: their counts are summed, the minimum
    cost is retained, and sampling probabilities are recomputed from the merged
    counts.

    Args:
        instance: The instance used to evaluate bitstring costs.
        solution: The solution to refine.

    Returns:
        A new solution with updated `bitstrings`, `costs`, `counts`, and `probabilities` reflecting the locally optimal results.
    """
    solution = deepcopy(solution)

    # If there are no bitstrings, return the solution unchanged.
    if solution.bitstrings.numel() == 0:
        return solution

    # Define an objective function that uses the existing evaluate_solution method.
    def qubo_objective(s_arr: Bitstring) -> float:
        # Convert the solution array to a list of integers
        return instance.evaluate_solution(s_arr)

    num_solutions = solution.bitstrings.shape[0]
    improved_bitstrings = bitstrings.zeros(num_solutions, instance.size)
    improved_costs = vector.zeros(num_solutions)

    for idx in range(num_solutions):
        # Get the current solution (row) as a numpy array of integers.
        s_orig = solution.bitstrings[idx].detach().clone()
        # Apply bit-flip local search to improve the solution.
        improved_bitstrings[idx, :], improved_costs[idx] = _bit_flip_local_search(
            qubo_objective, s_orig
        )

    unique_bitstrings, inverse = torch.unique(improved_bitstrings, dim=0, return_inverse=True)
    n = unique_bitstrings.shape[0]

    # Update the solution object.
    solution.bitstrings = unique_bitstrings
    solution.costs = vector.zeros(n).scatter_reduce(
        dim=0, index=inverse, src=improved_costs, reduce="amin", include_self=False
    )
    solution.counts = vectori.zeros(n).scatter_reduce(
        dim=0, index=inverse, src=solution.counts, reduce="sum", include_self=False
    )
    solution.compute_probabilities()

    return solution
