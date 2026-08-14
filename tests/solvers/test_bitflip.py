from __future__ import annotations

import pytest_check as check

from qubosolver import Solution, Instance, solvers, bitstrings, vectori, matrix


def test_solution_not_mutated() -> None:
    Q = matrix.tensor([[-1.0, 2.0], [2.0, -2.0]])
    instance = Instance(Q)

    solution = Solution(bitstrings.zeros(1, 2), counts=vectori.tensor([1]))
    solution._compute_costs(instance.matrix)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")

    new_solution = solvers.iterative_bitflip_local_search(instance, solution)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "00")
    check.equal(len(new_solution), 1)
    check.equal(new_solution[0].string, "01")
    check.is_not(new_solution, solution)

    new_solution2 = solvers.iterative_bitflip_local_search(instance, new_solution)
    check.equal(len(new_solution2), 1)
    check.equal(new_solution2[0].string, "01")
    check.is_not(new_solution2, new_solution)
