# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check

from qubosolver import Instance, Solution, solving

if TYPE_CHECKING:
    from types import ModuleType


def test_random() -> None:
    Q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    instance = Instance(matrix=Q)

    # Solve the QUBO problem.
    solution = solving.random_sampling.solve(instance, max_bitstrings=3)

    # Assert that the solution is an instance of Solution.
    check.is_instance(solution, Solution)
    check.equal(solution.num_variables, 2)  # two variables
    check.less_equal(len(solution), 3)
    check.is_true(solution.check_consistency(instance=instance, throw=True))


@pytest.mark.parametrize("solve_module", [solving.tabu_search, solving.simulated_annealing])
def test_time_limit(simple_qubo_instance: Instance, solve_module: ModuleType) -> None:
    # Use a very large iteration limit so that the solver is stopped
    # by the time limit rather than by max_iter.
    start_time = time.perf_counter()
    solution = solve_module.solve(simple_qubo_instance, max_iter=100_000_000, time_limit=0.01)
    elapsed_time = time.perf_counter() - start_time

    # Check that a valid solution is returned and that the solver
    # stops well before reaching the large iteration limit.
    assert isinstance(solution, Solution)
    assert elapsed_time < 1.0


@pytest.mark.parametrize(
    "solve_module",
    [
        solving.random_sampling,
        solving.tabu_search,
        solving.simulated_annealing,
        solving.trivial_solution_search,
        pytest.param(solving.cplex, marks=pytest.mark.extras),
        solving.brute_force,
        solving.iterative_bitflip_local_search,
    ],
    ids=[
        "random_sampling",
        "tabu_search",
        "simulated_annealing",
        "trivial_solution_search",
        "cplex",
        "brute_force",
        "iterative_bitflip_local_search",
    ],
)
def test_classical_solve_empty_instance(solve_module: ModuleType) -> None:
    instance = Instance()

    solution = solve_module.solve(instance)

    assert isinstance(solution, Solution)
    check.is_true(solution.check_consistency(instance=instance, throw=True))
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "")
    check.equal(solution[0].cost, 0.0)
    check.equal(solution[0].count, 1)
    check.equal(solution[0].probability, 1.0)
