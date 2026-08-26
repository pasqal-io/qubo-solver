from __future__ import annotations

import itertools
import numpy as np
import pytest
import pytest_check as check
import random
import torch

import qoolqit

from qubosolver import (
    Instance,
    Solution,
    SingleSolution,
    solvers,
    transforms,
    embedding,
    drive_shaping,
    torch_rng,
    bitstrings,
    vectori,
    vector,
    tensor,
    Tensor,
    Matrix,
)
from qubosolver.utils import analysis


def gather_optimal_solutions(solutions: Solution) -> list[SingleSolution]:
    min_cost = solutions[0].cost
    return [d for d in solutions if np.allclose(d.cost, min_cost)]


def interaction_matrix_from_vertices(vertices: Tensor) -> Matrix:
    U = 1.0 / torch.cdist(vertices, vertices) ** 6
    U.fill_diagonal_(0.0)
    return U


def simple_qubo() -> tuple[Instance, list[SingleSolution]]:

    sqrt3 = np.sqrt(3.0)
    vertices = tensor.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
            [4.0, 0.0],
        ],
    )
    print(vertices)
    d = torch.cdist(vertices, vertices, p=2)
    print(d)
    n_qubits = vertices.shape[0]
    diagonal_scale = -2.0
    diagonal = vector.zeros(n_qubits).fill_(1.0)
    Q = interaction_matrix_from_vertices(vertices) + diagonal_scale * torch.diag(diagonal)
    Q /= Q.max()

    solutions = Solution()
    solutions.bitstrings = bitstrings.tensor(list(itertools.product([0, 1], repeat=n_qubits)))
    solutions.counts = vectori.zeros(solutions.bitstrings.shape[0]).fill_(1)
    solutions._compute_costs(Q)._sort_by_cost()._compute_probabilities()

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(solutions)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    return Instance(matrix=Q), expected_optimal_solutions


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


def check_solution(
    solutions: Solution,
    expected_optimal_solutions: list[SingleSolution],
    *,
    expected_optimal_probability: float = 0.75,
) -> None:

    # Solutions are not duplicated
    check.equal(solutions.bitstrings.unique(dim=0).shape[0], solutions.bitstrings.shape[0])

    print(f"{analysis.to_dataframe([solutions])}")

    assert isinstance(solutions.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(solutions)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")

    if expected_optimal_probability == 0.0:
        return

    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)
    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for s in optimal_solutions:
        check.is_in(s.string, expected_optimal_bitstrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    check.greater(cumulated_probability, expected_optimal_probability)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_shaping_method", ["proportional_diagonal", "bayesian_search"])
@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
@pytest.mark.parametrize("postprocessing", [True, False], ids=["post", "no-post"])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no-pre"])
def test_quantum_solve(
    preprocessing: bool,
    postprocessing: bool,
    embedding_method: str,
    drive_shaping_method: str,
) -> None:

    seed = 16844214
    manual_seed(seed)
    qubo, expected_optimal_solutions = simple_qubo()

    device = qoolqit.AnalogDevice()

    # Try trivial solution (none here)
    trivial_solution = solvers.trivial_solution_search.solve(qubo)
    check.is_false(trivial_solution)

    effective_qubo = qubo
    if preprocessing:
        effective_qubo = transforms.variable_fixing.apply_recursively(qubo)

    if embedding_method == "blade":
        blade_config = embedding.blade.Config(device=device)
        register = embedding.blade.embed(effective_qubo, config=blade_config)
    elif embedding_method == "greedy_layout":
        greedy_config = embedding.greedy_layout.Config(traps=100)
        register = embedding.greedy_layout.embed(effective_qubo, device=device, config=greedy_config)
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method}")
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    emulator = qoolqit.execution.LocalEmulator()

    if drive_shaping_method == "proportional_diagonal":
        drive = drive_shaping.proportional_diagonal.build_drive(
            effective_qubo, register, device=device, dmm=False, kappa=0.25
        )
    elif drive_shaping_method == "bayesian_search":
        # Drive Bayesian Search is a hybrid solver that finds a solution and the
        # associated drive. Hence, it can be used as both a solver and a drive shaper
        _, drive = solvers.drive_bayesian_search.solve(
            effective_qubo,
            register,
            backend=emulator,
            device=device,
            dmm=False,
            config=solvers.drive_bayesian_search.Config(n_evaluations=11, seed=seed),
        )
    else:
        raise ValueError(f"Invalid drive shaping method: {drive_shaping_method}")

    job = solvers.analog_quantum_sampling.solve(register, drive, emulator, device)
    solution = Solution.from_results(job.results(), effective_qubo)

    # Post-process fixations of the preprocessing and restore the original QUBO
    if preprocessing:
        assert isinstance(effective_qubo, transforms.variable_fixing.Instance)
        solution = transforms.variable_fixing.lift(solution, effective_qubo)

    if postprocessing:
        solution = solvers.iterative_bitflip_local_search.solve(qubo, solution)

    expected_optimal_probability = 0.75
    if drive_shaping_method in ["bayesian_search"]:
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("solving_method", ["cplex", "tabu", "sa", "sa+tabu", "random"])
@pytest.mark.parametrize("postprocessing", [True, False], ids=["post", "no-post"])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no-pre"])
def test_classical_solve(
    preprocessing: bool,
    postprocessing: bool,
    solving_method: str,
) -> None:

    seed = 16844214
    rng = manual_seed(seed)
    qubo, expected_optimal_solutions = simple_qubo()

    # Try trivial solution (none here)
    trivial_solution = solvers.trivial_solution_search.solve(qubo)
    check.is_false(trivial_solution)

    effective_qubo = qubo
    if preprocessing:
        effective_qubo = transforms.variable_fixing.apply_recursively(qubo)

    if solving_method == "cplex":
        solution = solvers.cplex.solve(effective_qubo)
    elif solving_method == "tabu":
        solution = solvers.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=3)
        solution = solvers.tabu_search.solve(effective_qubo, solution.bitstrings)
    elif solving_method == "sa":
        solution = solvers.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=1)
        solution = solvers.simulated_annealing.solve(
            effective_qubo, solution[0].bitstring.unsqueeze(0), top_k=1
        )
    elif solving_method == "sa+tabu":
        solution = solvers.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=1)
        solution = solvers.simulated_annealing.solve(
            effective_qubo, solution[0].bitstring.unsqueeze(0), top_k=1
        )
        solution = solvers.tabu_search.solve(effective_qubo, solution.bitstrings)
    elif solving_method == "random":
        solution = solvers.random_sampling.solve(effective_qubo, rng=rng)
    else:
        raise ValueError(f"Invalid solving method: {solving_method}")

    if preprocessing:
        assert isinstance(effective_qubo, transforms.variable_fixing.Instance)
        solution = transforms.variable_fixing.lift(solution, effective_qubo)

    if postprocessing:
        solution = solvers.iterative_bitflip_local_search.solve(qubo, solution)

    expected_optimal_probability = 0.75
    if solving_method in ["random"]:
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )
