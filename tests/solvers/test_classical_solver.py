from __future__ import annotations

import pytest
import torch
import pytest_check as check
import itertools
import time
import random
import numpy as np

from qubosolver import (
    Instance,
    Solution,
    Solver,
    matrix,
    bitstring,
    torch_rng,
    SolverConfig,
    ClassicalSolvingConfig,
)
from qubosolver.solver._classical_solver import (
    get_classical_solver,
    SimulatedAnnealingSolver,
    TabuSearchSolver,
    RandomSolver,
)
from qubosolver.solver.config.solving import _ClassicalAlgorithm
from qubosolver.utils import _costs

class_solvers = {
    "simulated_annealing": SimulatedAnnealingSolver,
    "tabu_search": TabuSearchSolver,
}


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("classical_method", list(class_solvers.keys()))
@pytest.mark.parametrize("max_bitstrings", [1, 3])
def test_qubo_solver_sa_or_tabu(
    simple_qubo_instance: Instance, classical_method: _ClassicalAlgorithm, max_bitstrings: int
) -> None:

    seed = 1567
    manual_seed(seed)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalSolvingConfig(
        algorithm=classical_method,
        max_bitstrings=max_bitstrings,
        sa_seed=seed,
    )

    config = SolverConfig(solving=classical_config, activate_trivial_solutions=False)

    # insure get_classical_solver works properly
    check.is_instance(
        get_classical_solver(simple_qubo_instance, config.classical),
        class_solvers[classical_method],
    )

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = Solver(simple_qubo_instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of Solution.
    check.is_instance(solution, Solution)

    # assert shape matches config
    check.greater(len(solution), 0)
    check.less_equal(len(solution), max_bitstrings)  # max_bitstrings solution returned
    check.equal(solution.bitstrings.shape[1], simple_qubo_instance.size)
    check.equal(len(solution.counts), len(solution.bitstrings))
    check.equal(len(solution.probabilities), len(solution.bitstrings))
    # SA uses stats='per_run', so it counts each retained bitstring once,
    # bounded by max_bitstrings, same as tabu search.
    check.less_equal(solution.counts.sum().item(), max_bitstrings)
    check.almost_equal(solution.probabilities.sum().item(), 1.0)


def test_random() -> None:
    Q = matrix.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = Instance(matrix=Q)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalSolvingConfig(algorithm="random_sampling", max_bitstrings=3)
    config = SolverConfig(solving=classical_config, activate_trivial_solutions=False)

    # insure get_classical_solver works properly
    check.is_instance(get_classical_solver(instance, config.classical), RandomSolver)

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = Solver(instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of Solution.
    check.is_instance(solution, Solution)
    check.equal(solution.bitstrings.shape[1], 2)  # two variables
    check.less_equal(len(solution.bitstrings), classical_config.max_bitstrings)
    check.equal(len(solution.costs), len(solution.bitstrings))
    check.equal(len(solution.counts), len(solution.bitstrings))
    check.equal(len(solution.probabilities), len(solution.bitstrings))
    check.equal(solution.counts.sum().item(), classical_config.max_bitstrings)
    check.almost_equal(solution.probabilities.sum().item(), 1.0)


@pytest.mark.parametrize(
    "classical_methods",
    ["simulated_annealing"],
)
@pytest.mark.parametrize("max_bitstrings", [1])
def test_sa_cost(
    simple_qubo_instance: Instance, classical_methods: _ClassicalAlgorithm, max_bitstrings: int
) -> None:
    classical_config = ClassicalSolvingConfig(
        algorithm=classical_methods, max_bitstrings=max_bitstrings, sa_seed=42
    )

    config = SolverConfig(solving=classical_config, activate_trivial_solutions=False)

    check.is_instance(
        get_classical_solver(simple_qubo_instance, config.classical),
        class_solvers[classical_methods],
    )

    classical_solver = Solver(simple_qubo_instance, config)
    solution = classical_solver.solve()

    check.is_instance(solution, Solution)

    Q = simple_qubo_instance.matrix
    n = Q.shape[0]

    bitstrings = []
    costs_ = []

    for bits in itertools.product([0, 1], repeat=n):
        z = bitstring.tensor(bits)
        bitstrings.append(z)
        cost = _costs.quadratic_cost(z, Q)
        costs_.append(cost)

    sorted_results = sorted(zip(bitstrings, costs_), key=lambda x: x[1])
    bests = [(b, c) for b, c in sorted_results[:max_bitstrings]]

    for bitstring_, cost_ in bests:
        bitstring_sa = solution.bitstrings[0]
        cost_sa = solution.costs[0].item()

        torch.testing.assert_close(bitstring_, bitstring_sa)
        torch.testing.assert_close(cost_, cost_sa, rtol=1e-4, atol=0.0)


def test_tabu_time_limit(simple_qubo_instance: Instance) -> None:
    # Set max_iter and max_no_improve to very large values to ensure that
    # the solver is stopped by tabu_time_limit, not by another stop criterion.
    classical_config = ClassicalSolvingConfig(
        algorithm="tabu_search",
        max_bitstrings=1,
        max_iter=100_000_000,
        tabu_max_no_improve=100_000_000,
        tabu_time_limit=0.01,
    )

    config = SolverConfig(
        solving=classical_config,
        activate_trivial_solutions=False,
    )

    classical_solver = Solver(simple_qubo_instance, config)

    # Measure the full execution time of the solver.
    start_time = time.perf_counter()
    solution = classical_solver.solve()
    elapsed_time = time.perf_counter() - start_time

    # Check that a valid solution is returned and that the solver
    # stops well before reaching the large iteration limit.
    assert isinstance(solution, Solution)
    assert elapsed_time < 1.0


def test_sa_time_limit(simple_qubo_instance: Instance) -> None:
    # Use a very large iteration limit so that the solver is stopped
    # by the time limit rather than by max_iter.
    classical_config = ClassicalSolvingConfig(
        algorithm="simulated_annealing",
        max_bitstrings=1,
        max_iter=100_000_000,
        sa_time_limit=0.01,
    )

    config = SolverConfig(
        solving=classical_config,
        activate_trivial_solutions=False,
    )

    classical_solver = Solver(simple_qubo_instance, config)

    # Measure the full execution time of the solver.
    start_time = time.perf_counter()
    solution = classical_solver.solve()
    elapsed_time = time.perf_counter() - start_time

    # Check that a valid solution is returned and that the solver
    # stops well before reaching the large iteration limit.
    assert isinstance(solution, Solution)
    assert elapsed_time < 1.0


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "classical_method",
    [
        "simulated_annealing",
        "cplex",
        "tabu_search",
    ],
)
def test_empty_qubo_after_preprocessing(classical_method: _ClassicalAlgorithm) -> None:

    seed = 1846
    manual_seed(seed)

    # Use a very large iteration limit so that the solver is stopped
    # by the time limit rather than by max_iter.
    classical_config = ClassicalSolvingConfig(
        algorithm=classical_method,
        sa_seed=seed,
    )
    config = SolverConfig(
        solving=classical_config,
        preprocessing=True,
        activate_trivial_solutions=False,
    )

    instance = Instance(matrix=matrix.zeros(2))
    classical_solver = Solver(instance, config)

    solution = classical_solver.solve()
    solution._sort_by_cost()

    best_bitstring = bitstring.to_string(solution.bitstrings[0])
    check.equal(best_bitstring, "00")


if __name__ == "__main__":
    pytest.main()
