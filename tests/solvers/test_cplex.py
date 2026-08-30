from __future__ import annotations

import cplex
import pytest
import pytest_check as check
import torch

from qubosolver import (
    Instance,
    Solution,
    Solver,
    matrix,
    solving,
    vectori,
    tensor,
    torch_rng,
)
from qubosolver.solving.classical.cplex import _to_solution


def test_to_solution_without_incumbent() -> None:
    # Regression test flagged in review: if CPLEX's time/node limit is hit
    # before any incumbent is found, `problem.solution.get_values()` raises
    # an opaque `CplexSolverError: No solution exists`. `_to_solution` must
    # check `is_primal_feasible()` before extracting results and raise a
    # clear `RuntimeError` instead of letting that error propagate.
    #
    # Build the CPLEX problem directly (bypassing `_to_cplex`) and disable
    # presolve/heuristics/node exploration so CPLEX cannot find even the
    # trivial all-zero incumbent that an unconstrained QUBO always admits.
    n = 40
    Q = torch.rand(n, n, generator=torch_rng(1)) * 100 - 50
    Q = (Q + Q.T) / 2

    problem = cplex.Cplex()
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.variables.add(types="B" * n)
    problem.objective.set_quadratic(
        [cplex.SparsePair(ind=list(range(n)), val=Q[i].tolist()) for i in range(n)]
    )
    problem.parameters.optimalitytarget.set(3)  # accept non-convex QUBO objective
    problem.parameters.mip.limits.nodes.set(0)
    problem.parameters.mip.strategy.heuristicfreq.set(-1)
    problem.parameters.mip.strategy.probe.set(-1)
    problem.parameters.preprocessing.presolve.set(0)

    problem.solve()
    check.is_false(problem.solution.is_primal_feasible())

    with pytest.raises(RuntimeError):
        _to_solution(problem.solution)


def test_qubo_solver_classical_cplex() -> None:
    # Create a simple 2x2 QUBO instance.
    # For example, consider a QUBO where the optimum is known.
    # Here we use an identity matrix.
    Q = matrix.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = Instance(matrix=Q)

    # Create a solving.Config object with classical solver options.
    classical_config = solving.classical.Config(
        algorithm="cplex",
        cplex_maxtime=10.0,
        cplex_log_path="test_solver.log",
    )
    config = solving.Config(solving=classical_config)

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = Solver(instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of Solution.
    assert isinstance(solution, Solution)

    # Since we used an identity matrix as Q and the conversion in the CPLEX,
    #  solver multiplies the coefficients by 2,
    # the objective becomes 2*(x1 + x2).
    # The optimal value for binary variables is achieved when both are 0,
    # so expect a cost of 0.
    # Also, check that the bitstring has the expected shape, e.g., [1,2].
    assert solution.bitstrings.shape[0] == 1  # one solution returned
    assert solution.bitstrings.shape[1] == 2  # two variables

    # Additionally, check that the cost tensor is 0 (or very near to 0).
    expected_cost = 0.0
    actual_cost = solution.costs.item()  # convert cost tensor to a python float
    assert pytest.approx(actual_cost, rel=1e-3) == expected_cost


# cbfm-p_nvars040_inst004. Objective is x^T Q x over x in {0,1}^40, with Q
# symmetric. Diagonal below; couplings as (i, j, w) for the upper triangle,
# each mirrored into (j, i) -- a sparse encoding of a matrix that is 88% zeros.
# Every coefficient is an integer, so nothing here loses precision: the drift
# is created entirely inside CPLEX.
_ROUNDING_DIAGONAL = vectori.tensor(
    [
        -2,
        -8,
        -4,
        0,
        -10,
        -16,
        -16,
        -14,
        -14,
        -12,
        -2,
        -6,
        -8,
        -4,
        -4,
        -4,
        -4,
        -8,
        -2,
        -8,
        -6,
        -2,
        -4,
        -4,
        -10,
        -12,
        -2,
        -14,
        -6,
        -8,
        -8,
        -12,
        -8,
        -12,
        -6,
        -4,
        -2,
        -6,
        2,
        0,
    ]
)

_ROUNDING_COUPLINGS = tensor.tensor(
    [
        (1, 2, 2),
        (1, 31, 2),
        (1, 32, 2),
        (3, 30, 2),
        (3, 32, -2),
        (4, 5, 2),
        (4, 25, 2),
        (4, 29, 2),
        (4, 31, 2),
        (5, 25, 2),
        (5, 27, 2),
        (5, 29, 2),
        (5, 30, 2),
        (5, 31, 2),
        (5, 32, 2),
        (6, 7, 2),
        (6, 8, 2),
        (6, 19, 2),
        (6, 22, 2),
        (6, 25, 2),
        (6, 27, 2),
        (6, 29, 2),
        (7, 31, 2),
        (7, 32, 2),
        (7, 33, 2),
        (7, 36, 2),
        (7, 37, 2),
        (8, 19, 2),
        (8, 22, 2),
        (8, 24, 2),
        (8, 25, 2),
        (8, 27, 2),
        (8, 29, -2),
        (8, 30, 2),
        (9, 11, 2),
        (9, 25, 2),
        (9, 27, 2),
        (9, 29, 2),
        (9, 33, 2),
        (10, 36, -2),
        (10, 37, 2),
        (11, 33, 2),
        (12, 34, 2),
        (12, 35, 2),
        (12, 38, 2),
        (13, 24, 2),
        (14, 16, 2),
        (14, 33, 2),
        (14, 34, 2),
        (14, 38, -2),
        (15, 16, -2),
        (15, 20, 2),
        (15, 24, 2),
        (15, 26, 2),
        (15, 28, -2),
        (16, 33, 2),
        (16, 34, 2),
        (16, 35, 2),
        (16, 38, -2),
        (16, 39, -2),
        (17, 20, 2),
        (17, 26, 2),
        (17, 28, 2),
        (18, 26, -2),
        (18, 28, 2),
        (19, 20, 2),
        (23, 24, 2),
        (25, 26, -2),
        (25, 27, 2),
        (27, 28, 2),
        (31, 32, 2),
        (34, 35, -2),
        (36, 37, 2),
        (37, 38, -2),
    ],
    dtype=torch.int64,
)


def _build_rounding_matrix() -> Instance:
    Q = _ROUNDING_DIAGONAL.diag()
    i, j, w = _ROUNDING_COUPLINGS.unbind(dim=1)
    Q[i, j] = Q[j, i] = w
    return Instance(matrix.as_tensor(Q))


def test_rounding() -> None:
    # Regression test for issue #239: CPLEX's reported objective value can
    # drift from x^T Q x of the returned bitstring due to internal floating
    # point rounding. The Solution returned by the functional `cplex` solver
    # must report the actual cost of its own bitstring, not CPLEX's internal
    # (possibly rounded) objective value.
    instance = _build_rounding_matrix()
    solution = solving.cplex.solve(instance, maxtime=60.0)

    check.is_true(solution.check_consistency(instance=instance))

    best_cost = solution[0].cost
    expected_best_cost = solution._compute_costs(instance.matrix)[0].cost

    check.equal(best_cost, expected_best_cost)
