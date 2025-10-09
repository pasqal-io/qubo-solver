from __future__ import annotations

import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.utils.qubo_eval import qubo_cost


def qubo_simulated_annealing(
    qubo: QUBOInstance,
    max_iter: int = 100,
    initial_temp: float = 10.0,
    final_temp: float = 0.1,
    alpha: float = 0.99,
) -> QUBOSolution:
    """
    Solve a QUBO instance using the Simulated Annealing metaheuristic.

    This function wraps the low-level `simulated_annealing()` routine
    and converts its output into a standardized `QUBOSolution` object.

    The algorithm gradually lowers the system temperature to reduce
    the probability of accepting worse solutions, balancing exploration
    and exploitation.

    Args:
        qubo: The QUBO instance to optimize, providing the coefficient matrix
            and an evaluation method.
        max_iter: Maximum number of iterations to perform.
        initial_temp: Initial temperature at the start of annealing.
        final_temp: Minimum temperature at which the annealing process stops.
        alpha: Cooling rate applied at each iteration (0 < alpha < 1).

    Returns:
        A `QUBOSolution` object containing:
            - `bitstrings`: The best binary solution(s) found.
            - `costs`: Corresponding objective values as tensors.

    Example:
        >>> solution = qubo_simulated_annealing(qubo)
        >>> print(solution.bitstrings, solution.costs)
    """
    best_solution, _ = simulated_annealing(qubo, max_iter, initial_temp, final_temp, alpha)
    # add one dimension instead of rebuilding a ne tensor
    bitstrings = best_solution.unsqueeze(0).to(torch.float32)
    # bitstrings = torch.tensor([best_solution], dtype=torch.float32)
    costs = torch.tensor([qubo.evaluate_solution(best_solution.tolist())])
    return QUBOSolution(bitstrings=bitstrings, costs=costs)


def simulated_annealing(
    qubo: QUBOInstance,
    max_iter: int = 1000,
    initial_temp: float = 10.0,
    final_temp: float = 0.1,
    alpha: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Simulated Annealing optimization on a QUBO instance.

    The algorithm starts from a random binary solution and iteratively
    explores neighboring solutions by flipping one bit at a time. A new
    solution is accepted based on the Metropolis criterion:
    it is always accepted if it improves the objective, or with a probability
    proportional to `exp(-ΔE / T)` otherwise.

    The temperature decreases geometrically after each iteration, according
    to the cooling rate `alpha`.

    Args:
        qubo: The QUBO instance defining the cost matrix.
        max_iter: Maximum number of iterations to perform.
        initial_temp: Starting temperature (controls exploration).
        final_temp: Minimum temperature threshold for stopping.
        alpha: Cooling rate; should be slightly below 1 (e.g., 0.95–0.99).

    Returns:
        A tuple `(best_solution, best_energy)` where:
            - `best_solution`: Tensor of shape (n,) containing the best bitstring found.
            - `best_energy`: Scalar tensor with its corresponding cost.

    Example:
        >>> best_x, best_e = simulated_annealing(qubo)
        >>> print(best_x, best_e)
    """
    n = qubo.coefficients.shape[0]

    # Start with a random binary vector
    current_solution = torch.randint(low=0, high=2, size=(n,))
    current_energy = qubo_cost(current_solution, qubo.coefficients)

    best_solution = current_solution.clone()
    best_energy = current_energy

    temp = initial_temp
    iteration = 0

    while temp > final_temp and iteration < max_iter:
        # Propose a new solution by flipping one bit
        new_solution = current_solution.clone()
        flip_index = torch.randint(low=0, high=n - 1, size=(1,))
        new_solution[flip_index.data] ^= 1  # Bitflip

        new_energy = qubo_cost(new_solution, qubo.coefficients)
        delta = new_energy - current_energy

        # Decide whether to accept the new solution
        if delta < 0 or torch.rand(1) < torch.exp(-delta / temp):
            current_solution = new_solution
            current_energy = new_energy

            if current_energy < best_energy:
                best_solution = current_solution.clone()
                best_energy = current_energy

        temp *= alpha
        iteration += 1

    return best_solution, best_energy
