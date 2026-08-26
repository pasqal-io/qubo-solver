from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check

from qubosolver import (
    Instance,
    Solution,
    SingleSolution,
    solvers,
)

from qubosolver.utils import analysis

from qubos import QUBOS


def gather_optimal_solutions(solutions: Solution) -> list[SingleSolution]:
    min_cost = solutions[0].cost
    return [d for d in solutions if np.allclose(d.cost, min_cost)]


def check_solution(
    solution: Solution,
    qubo: Instance,
    expect_optimality: bool = True,
) -> float:

    # Solutions are not duplicated
    check.equal(solution.bitstrings.unique(dim=0).shape[0], len(solution))

    print(f"\n{analysis.to_dataframe([solution])}")

    optimal_solutions = gather_optimal_solutions(solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")

    if not expect_optimality:
        return 0.0

    expected_optimal_solutions = gather_optimal_solutions(
        solvers.brute_force.solve(qubo, max_bitstrings=-1)
    )
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)
    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for s in optimal_solutions:
        check.is_in(s.string, expected_optimal_bitstrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    return cumulated_probability


_n: int = len(QUBOS)


@pytest.mark.parametrize(
    "qubo_id",
    range(_n),
    ids=[f"qubo{i}" for i in range(_n)],
)
def test_cplex(
    qubo_id: int,
) -> None:
    instance = QUBOS[qubo_id]

    solution = solvers.cplex.solve(instance)
    check_solution(solution, instance)
