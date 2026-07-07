from __future__ import annotations

import itertools
import numpy as np
import pytest
import pytest_check as check
import random
import torch

import qoolqit

from qubosolver import (
    QUBOInstance,
    QUBOAnalyzer,
    QUBOSolver,
    SolverConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    ClassicalConfig,
    ClassicalSolverType,
    EmbedderType,
    DriveType,
    QUBOSingleSolution,
    QUBOSolution,
    bitstrings,
    vectori,
    torch_rng,
)


def gather_optimal_solutions(solutions: QUBOSolution) -> list[QUBOSingleSolution]:
    min_cost = solutions[0].cost
    return [d for d in solutions if np.allclose(d.cost, min_cost)]


def interaction_matrix_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
    U = 1.0 / torch.cdist(vertices, vertices) ** 6
    U.fill_diagonal_(0.0)
    return U


def simple_qubo() -> tuple[QUBOInstance, list[QUBOSingleSolution]]:

    sqrt3 = np.sqrt(3.0)
    vertices = torch.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    n_qubits = vertices.shape[0]
    diagonal_scale = -2.0
    diagonal = torch.ones(n_qubits, dtype=torch.float32)
    Q = interaction_matrix_from_vertices(vertices) + diagonal_scale * torch.diag(diagonal)
    Q /= Q.max()

    solutions = QUBOSolution()
    solutions.bitstrings = bitstrings.tensor(list(itertools.product([0, 1], repeat=n_qubits)))
    solutions.counts = vectori.zeros(solutions.bitstrings.shape[0]).fill_(1)
    solutions.compute_costs(Q).sort_by_cost().compute_probabilities()

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(solutions)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    return QUBOInstance(matrix=Q), expected_optimal_solutions


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


def check_solution(
    solutions: QUBOSolution,
    expected_optimal_solutions: list[QUBOSingleSolution],
    expect_optimality: bool = True,
) -> None:

    # Solutions are not duplicated
    check.equal(solutions.bitstrings.unique(dim=0).shape[0], solutions.bitstrings.shape[0])

    analyzer = QUBOAnalyzer([solutions])
    print(f"{analyzer.df}")

    assert isinstance(solutions.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(solutions)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")

    if not expect_optimality:
        return

    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)
    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for s in optimal_solutions:
        check.is_in(s.string, expected_optimal_bitstrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    check.greater(cumulated_probability, 0.75)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_shaping_method", ["heuristic", "optimized"])
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
        embedding_config = EmbeddingConfig(embedding_method=EmbedderType.BLADE, min_distance=1.001)
    elif embedding_method == "greedy":
        embedding_config = EmbeddingConfig(
            embedding_method=EmbedderType.GREEDY,
            min_distance=1.001,
            greedy_traps=100,
            greedy_spacing=0.1,
        )
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method}")

    if drive_shaping_method == "optimized":
        drive_shaping_config = DriveShapingConfig(
            drive_shaping_method=DriveType.OPTIMIZED,
            optimized_n_calls=11,
            optimized_seed=seed,
            dmm=False,
        )
    elif drive_shaping_method == "heuristic":
        drive_shaping_config = DriveShapingConfig(
            drive_shaping_method=DriveType.HEURISTIC, heuristic_kappa=0.25, dmm=False
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

    solver = QUBOSolver(qubo, config)
    solution = solver.solve()
    solution.compute_costs(qubo.matrix).sort_by_cost().compute_probabilities()

    register = solver._solver.embedding()
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    check_solution(solution, expected_optimal_solutions)


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
    manual_seed(seed)
    qubo, expected_optimal_solutions = simple_qubo()

    classical_solvers = {
        "cplex": ClassicalSolverType.CPLEX,
        "tabu": ClassicalSolverType.TABU_SEARCH,
        "sa": ClassicalSolverType.SIMULATED_ANNEALING,
        "sa+tabu": ClassicalSolverType.SIMULATED_ANNEALING_TABU_SEARCH,
        "random": ClassicalSolverType.RANDOM,
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

    solver = QUBOSolver(qubo, config)
    solution = solver.solve()
    solution.compute_costs(qubo.matrix).sort_by_cost().compute_probabilities()

    expect_optimality = solving_method not in ["random"]

    check_solution(solution, expected_optimal_solutions, expect_optimality)
