from __future__ import annotations

import cplex as CPLEX

from qubosolver import QUBOInstance, QUBOSolution, bitstrings, vector, vectori


def _qubo_instance_to_sparsepairs(
    instance: QUBOInstance, *, tol: float = 1e-8
) -> list[CPLEX.SparsePair]:
    """Convert a :class:`QUBOInstance` coefficient matrix to CPLEX sparse-pair format.

    Each row of the QUBO matrix is encoded as a :class:`cplex.SparsePair`,
    with coefficients scaled by 2 (CPLEX convention for quadratic objectives).

    Args:
        instance: The QUBO instance whose coefficients are converted.
        tol: Absolute tolerance below which coefficients are treated as zero.

    Returns:
        A list of :class:`cplex.SparsePair`, one per variable.
    """
    matrix = instance.matrix.cpu().numpy()
    size = matrix.shape[0]
    sparsepairs: list[CPLEX.SparsePair] = []

    for i in range(size):
        indices: list[int] = []
        values: list[float] = []
        for j in range(size):
            coeff = matrix[i, j] * 2
            if abs(coeff) > tol:
                indices.append(j)
                values.append(float(coeff))  # <<< conversion ici
        sparsepairs.append(CPLEX.SparsePair(ind=indices, val=values))

    return sparsepairs


def cplex(Q: QUBOInstance, *, maxtime: float = 600.0, log_path: str = "solver.log") -> QUBOSolution:
    """Solve a QUBO instance to optimality (or time limit) using IBM CPLEX.

    Args:
        Q: The :class:`QUBOInstance` to solve.
        maxtime: Maximum solver time in seconds. Defaults to 600.
        log_path: Path for the CPLEX log file. Defaults to ``"solver.log"``.

    Returns:
        A :class:`QUBOSolution` containing the optimal bitstring found by CPLEX.
    """
    # Determine the number of variables.
    N: int = Q.size
    # If there are no variables, return an empty solution.
    if N == 0:
        return QUBOSolution()

    # Convert the coefficient matrix into CPLEX sparse pairs format using the conversion tool.
    sparsepairs: list[CPLEX.SparsePair] = _qubo_instance_to_sparsepairs(Q)

    # Open a log file.
    log_file = open(log_path, "w")
    problem = CPLEX.Cplex()

    # Redirect logging streams.
    problem.set_log_stream(log_file)
    problem.set_error_stream(log_file)
    problem.set_warning_stream(log_file)
    problem.set_results_stream(log_file)

    problem.parameters.timelimit.set(maxtime)
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add binary variables.
    problem.variables.add(types="B" * N)

    # Set the quadratic objective.
    problem.objective.set_quadratic(sparsepairs)

    problem.solve()

    # Retrieve solution.
    solution_values = problem.solution.get_values()
    solution_cost = problem.solution.get_objective_value()

    log_file.close()

    # Convert the solution into a QUBOSolution.
    bitstring_tensor = bitstrings.tensor([[int(b) for b in solution_values]])
    counts = vectori.tensor([1])
    cost_tensor = vector.tensor([solution_cost])

    solution = QUBOSolution(
        bitstrings=bitstring_tensor, counts=counts, costs=cost_tensor
    ).compute_probabilities()
    return solution
