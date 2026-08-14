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
    Analyzer,
    Solver,
    SolverConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    ClassicalConfig,
    EmbedderType,
    SingleSolution,
    Solution,
    bitstrings,
    vectori,
    vector,
    tensor,
    Tensor,
    Matrix,
    torch_rng,
)
from qubosolver.solvers import ClassicalAlgorithm
from qubosolver.drive_shaping import Algorithm


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

    analyzer = Analyzer([solutions])
    print(f"{analyzer.df}")

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
@pytest.mark.parametrize("embedding_method", ["greedy", "blade"])
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

    if embedding_method == "blade":
        embedding_config = EmbeddingConfig(embedding_method=EmbedderType.BLADE)
    elif embedding_method == "greedy":
        embedding_config = EmbeddingConfig(
            embedding_method=EmbedderType.GREEDY,
            greedy_traps=100,
        )
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method}")

    if drive_shaping_method == "bayesian_search":
        drive_shaping_config = DriveShapingConfig(
            drive_shaping_method=Algorithm.BAYESIAN_SEARCH,
            bayesian_search_n_calls=11,
            bayesian_search_seed=seed,
            dmm=False,
        )
    elif drive_shaping_method == "proportional_diagonal":
        drive_shaping_config = DriveShapingConfig(
            drive_shaping_method=Algorithm.PROPORTIONAL_DIAGONAL,
            proportional_diagonal_kappa=0.25,
            dmm=False,
        )
    else:
        raise ValueError(f"Invalid drive shaping method: {drive_shaping_method}")

    config = SolverConfig(
        use_quantum=True,
        embedding=embedding_config,
        drive_shaping=drive_shaping_config,
        do_postprocessing=postprocessing,
        do_preprocessing=preprocessing,
        device=qoolqit.AnalogDevice(),
    )

    solver = Solver(qubo, config)
    solution = solver.solve()
    solution._compute_costs(qubo.matrix)._sort_by_cost()._compute_probabilities()

    register = solver._solver._embedding()
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    expected_optimal_probability = 0.75
    if drive_shaping_method in ["bayesian_search"]:
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("solving_method", ["cplex", "tabu", "sa", "random"])
@pytest.mark.parametrize("postprocessing", [True, False], ids=["post", "no-post"])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no-pre"])
def test_classical_solve(
    preprocessing: bool,
    postprocessing: bool,
    solving_method: str,
) -> None:

    seed = 16844214
    manual_seed(seed)
    qubo, expected_optimal_solutions = simple_qubo()

    classical_solvers = {
        "cplex": ClassicalAlgorithm.CPLEX,
        "tabu": ClassicalAlgorithm.TABU_SEARCH,
        "sa": ClassicalAlgorithm.SIMULATED_ANNEALING,
        "random": ClassicalAlgorithm.RANDOM,
    }

    classical_config = ClassicalConfig(
        classical_solver_type=classical_solvers[solving_method],
        max_bitstrings=1,
        sa_seed=seed,
    )

    config = SolverConfig(
        use_quantum=False,
        do_postprocessing=postprocessing,
        do_preprocessing=preprocessing,
        classical=classical_config,
    )

    solver = Solver(qubo, config)
    solution = solver.solve()
    solution._compute_costs(qubo.matrix)._sort_by_cost()._compute_probabilities()

    expected_optimal_probability = 0.75
    if solving_method in ["random"]:
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )
