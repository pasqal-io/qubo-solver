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

    Example:
        >>> solution = qubo_tabu_search(qubo, x0=torch.randint(0, 2, (10,)))
        >>> print(solution.costs)
    """
    best_solution, _ = tabu_search(
        qubo=qubo, x0=x0, max_iter=max_iter, tabu_tenure=tabu_tenure, max_no_improve=max_no_improve
    )
    bitstrings = best_solution.unsqueeze(0).to(torch.float32)
    costs = torch.tensor([qubo.evaluate_solution(best_solution.tolist())], dtype=torch.float32)
    return QUBOSolution(bitstrings=bitstrings, costs=costs)


def tabu_search(
    qubo: QUBOInstance,
    x0: torch.Tensor,
    max_iter: int = 100,
    tabu_tenure: int = 7,
    max_no_improve: int = 20,
) -> tuple[torch.Tensor, float]:
    """
    Perform Tabu Search on a QUBO instance to find a low-cost bitstring.

    The algorithm iteratively flips bits in the current solution to
    explore neighboring solutions, while maintaining a tabu list to
    prevent cycling. It keeps track of the best solution encountered
    and stops when no improvement is observed for a given number of
    iterations.

    Args:
        qubo: The QUBO instance providing the cost matrix.
        x0: The initial binary solution tensor of shape (n,).
        max_iter: Maximum number of search iterations.
        tabu_tenure: Number of iterations a move (bit flip) remains tabu.
        max_no_improve: Maximum number of consecutive iterations
            without improvement before termination.

    Returns:
        A tuple `(x_best, f_best)` where:
            - `x_best`: Tensor representing the best bitstring found.
            - `f_best`: Corresponding objective value as a float.

    Example:
        >>> best_x, best_cost = tabu_search(qubo, torch.randint(0, 2, (10,)))
        >>> print(best_x, best_cost)
    """
    Q = qubo.coefficients
    n: int = x0.numel()

    x_best = x0.clone().to(torch.int64)
    f_best: float = qubo_cost(x_best, Q).item()

    x_current = x0.clone()
    f_current: float = f_best

    # Tabu list: store iteration number until which each move is tabu
    tabu_list = torch.zeros(n)
    iter_since_last_improve: int = 0

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

        # Update best solution if improved
        if f_current < f_best:
            x_best = x_current.clone()
            f_best = f_current
            iter_since_last_improve = 0
        else:
            iter_since_last_improve += 1

        if iter_since_last_improve >= max_no_improve:
            break  # Stop if no improvement for a while

    return x_best, f_best
