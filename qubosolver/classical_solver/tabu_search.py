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
    max_bitstrings: int = 1,
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
            - `probabilities`: Frequencies divided by max_bitstrings.

    Example:
        >>> solution = qubo_tabu_search(qubo, x0=torch.randint(0, 2, (10,)))
        >>> print(solution)
    """
    best_solutions, costs, counts = tabu_search(
        qubo=qubo,
        x0=x0,
        max_iter=max_iter,
        tabu_tenure=tabu_tenure,
        max_no_improve=max_no_improve,
        max_bitstrings=max_bitstrings,
    )
    return QUBOSolution(
        bitstrings=best_solutions,
        costs=costs,
        counts=counts,
        probabilities=counts.float() / max_bitstrings,
    )


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
    n = x0.numel()

    # Repeat x0 for each parallel run
    x_current = x0.clone().to(torch.int64).unsqueeze(0).repeat(max_bitstrings, 1)
    f_current = qubo_cost(x_current, Q)
    x_best = x_current.clone()
    f_best = f_current.clone()

    # Tabu list per run and bit
    tabu_list = torch.zeros((max_bitstrings, n), dtype=torch.int64, device=device)
    iter_since_last_improve = torch.zeros(max_bitstrings, dtype=torch.int64, device=device)
    for iteration in range(max_iter):
        # Generate all neighbor candidates for each bit flip
        flips = torch.eye(n, dtype=torch.int64, device=device).unsqueeze(0)
        x_neighbors = x_current.unsqueeze(1).clone()
        x_neighbors = (x_neighbors + flips) % 2  # each bit flipped
        f_candidates = qubo_cost(x_neighbors.view(-1, n), Q).view(max_bitstrings, n)

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
    costs = qubo_cost(uniq, Q)

    return uniq, costs, counts
