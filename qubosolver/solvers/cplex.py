"""QUBO solver backed by IBM CPLEX.

Formulates the QUBO problem as a Binary Quadratic Program (BQP) and solves it
with IBM CPLEX's branch-and-bound MIP engine, which guarantees an optimal
solution within the given time limit.

.. note::
    This module requires the ``cplex`` Python package (part of IBM CPLEX
    Optimization Studio) to be installed.  It is an optional dependency; the
    rest of ``qubosolver`` works without it.  A missing ``cplex`` package will
    raise :exc:`ModuleNotFoundError` at import time.

Typical usage goes through :class:`~qubosolver.solvers.CplexSolver`, which
reads :class:`~qubosolver.config.ClassicalConfig` parameters and calls
`cplex` directly.
"""

from __future__ import annotations

import cplex as CPLEX

from qubosolver import Instance, Solution, bitstrings, vector, vectori


def _qubo_instance_to_sparsepairs(
    instance: Instance, *, tol: float = 1e-8
) -> list[CPLEX.SparsePair]:
    """Convert a :class:`Instance` coefficient matrix to CPLEX sparse-pair format.

    CPLEX evaluates quadratic objectives as ``½ · xᵀ Q_cplex x``, so each
    coefficient must be pre-multiplied by 2 to recover the standard QUBO
    objective ``xᵀ Q x``.

    Near-zero coefficients (``|coeff * 2| <= tol``) are dropped to keep the
    sparse representation compact and avoid numerical noise.

    Args:
        instance: The QUBO instance whose coefficient matrix is converted.
            The matrix is moved to CPU and cast to a NumPy array before
            processing.
        tol: Absolute threshold for dropping small coefficients after the ×2
            scaling.  Defaults to ``1e-8``.

    Returns:
        A list of :class:`cplex.SparsePair` of length ``instance.size``,
        where element *i* encodes the non-zero scaled coefficients in row *i*
        of the QUBO matrix.
    """
    matrix = instance.matrix.cpu().numpy()
    size = matrix.shape[0]
    sparsepairs: list[CPLEX.SparsePair] = []

    for i in range(size):
        indices: list[int] = []
        values: list[float] = []
        for j in range(size):
            coeff = matrix[i, j] * 2  # scale by 2 to cancel CPLEX's ½ factor
            if abs(coeff) > tol:
                indices.append(j)
                values.append(float(coeff))
        sparsepairs.append(CPLEX.SparsePair(ind=indices, val=values))

    return sparsepairs


def cplex(instance: Instance, *, maxtime: float = 600.0, log_path: str = "") -> Solution:
    """Solve a QUBO instance to optimality (or time limit) using IBM CPLEX.

    Builds a Binary Quadratic Program from *instance*, sets a time limit, and runs
    CPLEX's branch-and-bound solver.  All CPLEX log output (progress,
    warnings, errors) is redirected to *log_path*, which is opened in write
    mode (overwriting any existing file) and closed after solving.

    Returns an empty :class:`~qubosolver.types.Solution` immediately when
    ``instance.size == 0`` without invoking CPLEX.

    Args:
        instance: The :class:`~qubosolver.types.Instance` to solve.
        maxtime: Wall-clock time limit for CPLEX in seconds.  CPLEX returns
            the best feasible solution found so far when the limit is reached.
            Defaults to ``600.0``.
        log_path: File path where CPLEX log output (progress, warnings,
            errors) is written, opened in write mode (``"w"``), so any
            existing file is overwritten.  When empty (the default), logging
            is suppressed and no file is created.

    Returns:
        A :class:`~qubosolver.types.Solution` containing exactly one
        bitstring — the best (or optimal) solution found by CPLEX — with
        ``count=1`` and ``probability=1.0``.  Returns an empty
        :class:`~qubosolver.types.Solution` if ``instance.size == 0``.

    Raises:
        cplex.exceptions.CplexError: If CPLEX encounters an internal solver
            error (e.g. infeasible model or licence issue).
    """
    # Determine the number of variables.
    N: int = instance.size
    # If there are no variables, return an empty solution.
    if N == 0:
        return Solution()

    # Convert the coefficient matrix into CPLEX sparse pairs format using the conversion tool.
    sparsepairs: list[CPLEX.SparsePair] = _qubo_instance_to_sparsepairs(instance)

    problem = CPLEX.Cplex()

    if log_path:
    # Open a log file.
        log_file = open(log_path, "w")

        # Redirect logging streams.
        problem.set_log_stream(log_file)
        problem.set_error_stream(log_file)
        problem.set_warning_stream(log_file)
        problem.set_results_stream(log_file)
    else:
        log_file = None

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

    if log_file:
        log_file.close()

    # Convert the solution into a Solution.
    bitstring_tensor = bitstrings.tensor([[int(b) for b in solution_values]])
    counts = vectori.tensor([1])
    cost_tensor = vector.tensor([solution_cost])

    solution = Solution(
        bitstrings=bitstring_tensor, counts=counts, costs=cost_tensor
    ).compute_probabilities()
    return solution
