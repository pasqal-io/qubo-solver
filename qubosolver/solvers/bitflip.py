from __future__ import annotations

import torch
from collections.abc import Callable


from qubosolver import QUBOInstance, QUBOSolution, bitstrings, vector, vectori, Bitstring


def _bit_flip_local_search(
    qubo_func: Callable[[Bitstring], float],
    s: Bitstring,
    rng: torch.Generator | None = None,
) -> tuple[Bitstring, float]:
    """
    Performs a local search by flipping bits to improve the objective value.

    Args:
        qubo_func: Function that computes the objective value for a solution array.
        s (np.ndarray): Binary array representing a candidate solution.
        shuffle (bool, optional): Shuffle to diversify

    Returns:
        tuple[np.ndarray, float]: The improved solution and its objective value.
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


def iterative_bitflip_local_search(Q: QUBOInstance, solution: QUBOSolution) -> QUBOSolution:
    """Improve every bitstring in *solution* via greedy single-bit-flip local search.

    Each bitstring is independently refined by flipping the bit that yields the
    largest cost decrease, repeating until no single flip improves the cost.
    Duplicate bitstrings are merged after improvement.

    Args:
        solution: The initial :class:`QUBOSolution` to refine.
        Q: The :class:`QUBOInstance` used for cost evaluation.

    Returns:
        The refined :class:`QUBOSolution` with improved (and deduplicated) bitstrings.
    """
    # If there are no bitstrings, return the solution unchanged.
    if solution.bitstrings.numel() == 0:
        return solution

    # Define an objective function that uses the existing evaluate_solution method.
    def qubo_objective(s_arr: Bitstring) -> float:
        # Convert the solution array to a list of integers
        return Q.evaluate_solution(s_arr)

    num_solutions = solution.bitstrings.shape[0]
    improved_bitstrings = bitstrings.zeros(num_solutions, Q.size)
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
