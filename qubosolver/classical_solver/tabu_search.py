from __future__ import annotations

import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.utils.qubo_eval import qubo_cost


def qubo_tabu_search(
    qubo: QUBOInstance,
    x0: torch.Tensor,
    max_iter: int = 100,
    tabu_tenure: int = 7,
    max_no_improve: int = 20,
) -> QUBOSolution:
    """
    Solve a QUBO problem using a simple Tabu Search heuristic.

    This function wraps the core `tabu_search()` routine and converts
    its output into a standardized `QUBOSolution` object, including
    the bitstrings and their evaluated costs.

    Args:
        qubo: The QUBO instance to optimize, providing the cost matrix
            and an evaluation method.
        x0: The initial solution as a binary tensor of shape (n,).
        max_iter: Maximum number of iterations to perform.
        tabu_tenure: Number of iterations a flipped variable remains tabu.
        max_no_improve: Stop criterion based on consecutive iterations
            without improvement.

    Returns:
        A `QUBOSolution` object containing:
            - `bitstrings`: The best solution found as a tensor.
            - `costs`: The corresponding objective value tensor.
            - `counts`: The frequencies of each bitstring.

    Example:
        >>> solution = qubo_tabu_search(qubo, x0=torch.randint(0, 2, (10,)))
        >>> print(solution.costs)
    """
    best_solutions, costs, counts = tabu_search(
        qubo=qubo, x0=x0, max_iter=max_iter, tabu_tenure=tabu_tenure, max_no_improve=max_no_improve
    )
    return QUBOSolution(bitstrings=best_solutions, costs=costs, counts=counts)


def tabu_search(
    qubo: QUBOInstance,
    x0: torch.Tensor,
    max_iter: int = 100,
    tabu_tenure: int = 7,
    max_no_improve: int = 20,
    max_bitstrings: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Tabu Search on a QUBO instance to find low-cost bitstrings.

    The algorithm iteratively flips bits in the current solution to
    explore neighboring solutions, while maintaining a tabu list to
    prevent cycling. It keeps track of the best solution encountered
    and stops when no improvement is observed for a given number of
    iterations.

    Args:
        qubo (QUBOInstance): The QUBO instance providing the cost matrix.
        x0 (torch.Tensor): The initial binary solution tensor of shape (n,).
        max_iter (int, optional): Maximum number of search iterations.
        tabu_tenure (int, optional): Number of iterations a move (bit flip)
            remains tabu. Defaults to 7.
        max_no_improve (int, optional): Maximum number of consecutive iterations
            without improvement before termination. Defaults to 20.
        max_bitstrings (int, optional): Maximum number of bitstring solutions returned.
            Defaults to 1.

    Returns:
        A tuple `(bistrings, costs, counts)` where:
            - `bistrings`: Tensor representing the best bitstrings found.
            - `costs`: Corresponding objective values.
            - `counts`: Frequencies each bitstring was found.

    Example:
        >>> x, costs, counts = tabu_search(qubo, torch.randint(0, 2, (10,)))
        >>> print(x, costs, counts)
    """
    Q = qubo.coefficients
    device = Q.device
    n: int = x0.numel()

    x_best = x0.clone().to(torch.int64)
    f_best = qubo_cost(x_best, Q).item()

    x_current = x0.clone()
    f_current = f_best

    # Tabu list: store iteration number until which each move is tabu
    tabu_list = torch.zeros(n)
    iter_since_last_improve: int = 0

    # Fixed-size buffers for at most max_bitstrings solutions
    bitstrings = torch.zeros((max_bitstrings, n), dtype=torch.int64, device=device)
    costs = torch.full((max_bitstrings,), float("inf"), device=device)
    counts = torch.zeros((max_bitstrings,), dtype=torch.int64, device=device)
    num_stored = 0

    def add_solution(x: torch.Tensor, cost: torch.Tensor) -> None:
        nonlocal num_stored
        # Check if already stored
        if num_stored > 0:
            eq_mask = torch.all(bitstrings[:num_stored] == x, dim=1)
            if torch.any(eq_mask):
                idx = torch.nonzero(eq_mask, as_tuple=False)[0].item()
                counts[idx] += 1
                return
        # Otherwise add or replace
        if num_stored < max_bitstrings:
            bitstrings[num_stored] = x
            costs[num_stored] = cost
            counts[num_stored] = 1
            num_stored += 1
        else:
            worst_idx = torch.argmax(costs)
            if cost < costs[worst_idx]:
                bitstrings[worst_idx] = x
                costs[worst_idx] = cost
                counts[worst_idx] = 1

    # Add starting solution
    add_solution(x_current, f_current)

    for iteration in range(max_iter):
        best_candidate = None
        best_candidate_cost = torch.inf
        best_move = -1

        for i in range(n):
            x_candidate = x_current.clone()
            x_candidate[i] = 1 - x_candidate[i]  # Bitflip
            f_candidate: float = qubo_cost(x_candidate, Q).item()

            # Check if move is tabu OR aspiration criterion (better than best)
            if tabu_list[i] <= iteration or f_candidate < f_best:
                if f_candidate < best_candidate_cost:
                    best_candidate = x_candidate
                    best_candidate_cost = f_candidate
                    best_move = i

        if best_candidate is None:
            break  # No valid move found

        # Apply best move
        x_current = best_candidate.clone()
        f_current = best_candidate_cost
        tabu_list[best_move] = iteration + tabu_tenure

        add_solution(x_current, f_current)

        # Update best solution if improved
        if f_current < f_best:
            x_best = x_current.clone()
            f_best = f_current
            iter_since_last_improve = 0
        else:
            iter_since_last_improve += 1

        if iter_since_last_improve >= max_no_improve:
            break  # Stop if no improvement for a while

    # Trim unused buffer slots
    bitstrings = bitstrings[:num_stored]
    costs = costs[:num_stored]
    counts = counts[:num_stored]

    # Sort by cost ascending
    sorted_idx = torch.argsort(costs, descending=True)
    bitstrings = bitstrings[sorted_idx]
    costs = costs[sorted_idx]
    counts = counts[sorted_idx]

    return bitstrings, costs, counts
