"""QUBO solver backed by IBM CPLEX.

Formulates the QUBO problem as a Binary Quadratic Program (BQP) and solves it
with IBM CPLEX's branch-and-bound MIP engine, which guarantees an optimal
solution within the given time limit.

Note:
    This module requires the ``cplex`` Python package (part of IBM CPLEX
    Optimization Studio) to be installed. Install it with the ``extras``
    extra: ``pip install 'qubo-solver[extras]'``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from qubosolver import Instance, Solution, bitstrings, vector, vectori

if TYPE_CHECKING:
    import cplex as CPLEX


def _import_cplex() -> Any:  # noqa: ANN401 (dynamically imported optional module)
    """Import and return the ``cplex`` module, with a helpful error if absent.

    Raises:
        ImportError: If the ``cplex`` package is not installed.
    """
    try:
        import cplex

        return cplex
    except ImportError as e:
        raise ImportError(
            "Solving with CPLEX requires the 'cplex' package. Install it with: "
            "pip install 'qubo-solver[extras]'"
        ) from e


def _qubo_instance_to_sparsepairs(
    instance: Instance, *, tol: float = 1e-8
) -> list[CPLEX.SparsePair]:
    r"""Convert an [`Instance`][] coefficient matrix to CPLEX sparse-pair format.

    CPLEX evaluates quadratic objectives as $\\frac{1}{2} x^T Q_{cplex} x$, so
    each coefficient must be pre-multiplied by 2 to recover the standard QUBO
    objective $x^T Q x$.

    Near-zero coefficients (``|coeff * 2| <= tol``) are dropped to keep the
    sparse representation compact and avoid numerical noise.

    Args:
        instance: The QUBO instance whose coefficient matrix is converted.
            The matrix is moved to CPU and cast to a NumPy array before
            processing.
        tol: Absolute threshold for dropping small coefficients after the
            x2 scaling.

    Returns:
        A list of `cplex.SparsePair` of length ``instance.size``, where
            element *i* encodes the non-zero scaled coefficients in row *i* of
            the QUBO matrix.
    """
    cplex_module = _import_cplex()

    size = instance.size
    sparsepairs: list[CPLEX.SparsePair] = []
    matrix = instance.matrix.cpu().numpy()

    for i in range(size):
        indices: list[int] = []
        values: list[float] = []
        for j in range(size):
            coeff = matrix[i, j] * 2  # scale by 2 to cancel CPLEX's ½ factor
            if abs(coeff) > tol:
                indices.append(j)
                values.append(float(coeff))
        sparsepairs.append(cplex_module.SparsePair(ind=indices, val=values))

    return sparsepairs


def _to_cplex(
    instance: Instance,
    *,
    log_file: Any = None,  # noqa: ANN401 (file-like object forwarded to CPLEX's log streams)
) -> CPLEX.Cplex:
    """Build the minimal CPLEX problem representing a QUBO instance.

    Sets only what is needed to represent the QUBO instance as a CPLEX
    problem (binary variables, minimization sense, quadratic objective), plus
    logging streams as the sole exception. Every other parameter, including
    the time limit, must be set by the caller on the returned problem.

    Args:
        instance: The QUBO instance to translate into a CPLEX problem.
        log_file: File-like object (or `None`) passed to CPLEX's log, error,
            warning, and results streams.

    Returns:
        A `cplex.Cplex` problem with binary variables, minimization sense,
            and the quadratic objective set, ready for the caller to
            configure further (e.g. time limit) and solve.
    """
    cplex_module = _import_cplex()

    # Convert the coefficient matrix into CPLEX sparse pairs format using the conversion tool.
    sparsepairs: list[CPLEX.SparsePair] = _qubo_instance_to_sparsepairs(instance)

    problem = cplex_module.Cplex()

    # Redirect logging streams.
    problem.set_log_stream(log_file)
    problem.set_error_stream(log_file)
    problem.set_warning_stream(log_file)
    problem.set_results_stream(log_file)

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add binary variables.
    problem.variables.add(types="B" * instance.size)

    # Set the quadratic objective.
    problem.objective.set_quadratic(sparsepairs)

    return problem


def _to_solution(cplex_solution: CPLEX.SolutionInterface) -> Solution:
    """Extract a [`Solution`][] from a solved CPLEX solution interface.

    Args:
        cplex_solution: The solution interface of a CPLEX problem that has
            already been solved.

    Returns:
        A [`Solution`][] holding the single incumbent bitstring, its cost,
            and a count of 1, with probabilities computed.

    Raises:
        RuntimeError: If CPLEX has no incumbent to report (e.g. the time or
            node limit was reached before any feasible solution was found),
            since `get_values`/`get_objective_value` raise an opaque
            `CplexSolverError` in that case.
    """
    if not cplex_solution.is_primal_feasible():
        raise RuntimeError("CPLEX found no feasible solution within the given time/node limit.")

    solution_values = cplex_solution.get_values()
    solution_cost = cplex_solution.get_objective_value()

    # Convert the solution into a Solution.
    # CPLEX's default integrality tolerance is 1e-5, looser than `round`'s default.
    bitstring_tensor = bitstrings.round([solution_values], atol=1e-4)
    counts = vectori.tensor([1])
    cost_tensor = vector.tensor([solution_cost])

    solution = Solution(
        bitstrings=bitstring_tensor, counts=counts, costs=cost_tensor
    )._compute_probabilities()
    return solution


def solve(instance: Instance, *, maxtime: float = 600.0, log_path: str = "") -> Solution:
    """Solve a QUBO instance to optimality (or time limit) using IBM CPLEX.

    Args:
        instance: The QUBO instance to solve.
        maxtime: Wall-clock time limit for CPLEX in seconds. CPLEX returns
            the best feasible solution found so far when the limit is
            reached.
        log_path: File path where CPLEX log output (progress, warnings,
            errors) is written, opened in write mode (``"w"``), so any
            existing file is overwritten. When empty (the default), logging
            is suppressed and no file is created.

    Returns:
        A solution containing exactly one bitstring — the best (or
            optimal) solution found by CPLEX.
    """
    # If there are no variables, return an empty solution.
    if not instance:
        return Solution()

    # Open a log file, or a no-op context manager if none was requested.
    with open(log_path, "w") if log_path else contextlib.nullcontext() as log_file:
        problem = _to_cplex(instance, log_file=log_file)
        problem.parameters.timelimit.set(maxtime)

        problem.solve()

        # Retrieve solution.
        solution = _to_solution(problem.solution)

    return solution
