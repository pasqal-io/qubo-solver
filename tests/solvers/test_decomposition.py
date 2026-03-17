from __future__ import annotations

import pytest
import pytest_check as check
import torch
import itertools
import numpy as np
import random
from typing import Tuple

from qoolqit.devices import DigitalAnalogDevice
from qubosolver import QUBOInstance

from qubosolver.config import SolverConfig, DecompositionConfig
from qubosolver.solver import DecomposeQuboSolver, QuboSolver
from qubosolver.data import QUBODataset
from qubosolver.algorithms.decompose import compute_distance_interaction_matrix


@pytest.mark.parametrize("use_quantum", [True, False])
def test_initial_steps_solver(decomposable_qubo: QUBOInstance, use_quantum: bool) -> None:
    """Test that the first steps of the decomposition (initialization +
    one loop iteration of a decomposition) are yielding corrent tensors
    or dictionaries of right sizes.

    """
    from qubosolver.algorithms.decompose import (
        compute_distance_interaction_matrix,
        geometric_search,
        interaction_matrix_from_placed,
        transfer_edge_values,
        update_global_solution,
        vertices_to_place,
    )

    size = decomposable_qubo.size
    qubo_mat = decomposable_qubo.coefficients.clone()

    decompose_config = DecompositionConfig()
    config = SolverConfig(use_quantum=use_quantum, decompose=decompose_config)
    solver = QuboSolver(decomposable_qubo, config)

    ## Check the distance interaction matrix matches the qubo matrix
    dist_matrix = compute_distance_interaction_matrix(
        solver._solver.device._pulser_device, qubo_mat
    )
    assert dist_matrix.shape == qubo_mat.shape
    assert torch.all(torch.diag(dist_matrix) == torch.diag(qubo_mat))

    ## Check for the dictionary of vertices to place, dimensions are correct
    current_vertices_dict = vertices_to_place(dist_matrix, qubo_mat)
    assert len(current_vertices_dict) == size
    for i in current_vertices_dict.keys():
        assert len(current_vertices_dict[i]["blocking_vertices"]) <= size
        assert len(current_vertices_dict[i]["separated_vertices"]) <= size
        assert len(current_vertices_dict[i]["neighbors_id"]) <= size
        assert len(current_vertices_dict[i]["neighbors_id"]) == len(
            current_vertices_dict[i]["neighbors_weight"]
        )

    # check that the initial transfer does not affect the length of the dictionary.
    solution = torch.full((size,), -1)
    transfer_edge_values(current_vertices_dict, dict(), solution, qubo_mat)
    assert len(current_vertices_dict) == size

    # try one iteration, check placed_vertices length
    config_subproblems = config.model_copy(update={"decompose": False})
    first_vertex = 0

    placed_vertices = geometric_search(
        qubo_mat,
        current_vertices_dict,
        first_vertex,
        decompose_config.decompose_threshold,
        solver._solver.device._pulser_device,
    )
    assert len(placed_vertices) <= size

    # check matrix size correspond to placed_vertices
    matrix_to_solve, map_index_vertices = interaction_matrix_from_placed(
        placed_vertices, solver._solver.device._pulser_device
    )
    assert len(map_index_vertices) == len(placed_vertices) == matrix_to_solve.shape[0]
    subproblem = QUBOInstance(matrix_to_solve)
    subsolver = solver._solver._solver_factory(  # type: ignore[attr-defined]
        subproblem, config_subproblems
    )
    sub_solution = subsolver.solve().bitstrings[0]

    # test update_global_solution remove -1 values
    update_global_solution(
        global_solution=solution, sub_solution=sub_solution, mapping=map_index_vertices
    )
    assert (solution == -1).sum() < size

    # test the transfer changes current_vertices_dict with less vertices to place
    transfer_edge_values(current_vertices_dict, placed_vertices, solution, qubo_mat)
    assert len(current_vertices_dict) < size


@pytest.mark.parametrize("use_quantum", [True, False])
def test_decomp_solver(decomposable_qubo: QUBOInstance, use_quantum: bool) -> None:
    config = SolverConfig(use_quantum=use_quantum, decompose=DecompositionConfig())
    solver = QuboSolver(decomposable_qubo, config)

    assert isinstance(solver._solver, DecomposeQuboSolver)

    solution = solver.solve()

    # check that only one solution is returned
    assert solution.counts is not None
    assert solution.counts.sum() == 1
    assert len(solution.bitstrings) == 1
    assert (solution.bitstrings[0] == -1).sum() == 0
    assert solution.costs.item() <= 0

    # check that many iterations were done
    assert solver._solver.number_iterations >= 0


def test_small_qubo_solver(simple_qubo_instance: QUBOInstance) -> None:

    # assert that the decomposition falls back to not being used as qubo is small
    simple_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(use_quantum=False, decompose=None),
    )
    solutions1 = simple_solver.solve()

    decompose_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(use_quantum=False, decompose=DecompositionConfig()),
    )
    solutions2 = decompose_solver.solve()
    assert isinstance(decompose_solver._solver, DecomposeQuboSolver)
    assert decompose_solver._solver.number_iterations == 0

    assert torch.allclose(solutions2.costs.min(), solutions1.costs.min())


def test_scope(decomposable_qubo: QUBOInstance) -> None:

    config = SolverConfig(use_quantum=False, decompose=DecompositionConfig())

    # check negative off-diagonal are not supported
    coeffs = decomposable_qubo.coefficients
    coeffs[0][1] = -1.0

    with pytest.raises(
        ValueError, match="Decomposition does not handle off-diagonal negative coefficients"
    ):
        QuboSolver(QUBOInstance(coeffs), config)


def test_compute_distance_interaction_matrix_zero_output() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0
    device = DigitalAnalogDevice()

    Q = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
        dtype=torch.float32,
    )

    dist_matrix = compute_distance_interaction_matrix(
        device._pulser_device, Q, neglecting_inter_distance, neglecting_max_coefficient
    )

    torch.testing.assert_close(dist_matrix, torch.zeros_like(Q))


def test_compute_distance_interaction_diagonal() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0
    device = DigitalAnalogDevice()

    Q = torch.tensor(
        [
            [-10, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    dist_matrix = compute_distance_interaction_matrix(
        device._pulser_device, Q, neglecting_inter_distance, neglecting_max_coefficient
    )
    expected_dist_matrix = neglecting_inter_distance * torch.ones_like(Q)
    expected_dist_matrix.diagonal().copy_(Q.diag())

    torch.testing.assert_close(dist_matrix, expected_dist_matrix)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("dims", [(4,), (3,), (3, 3), (2, 3, 2), (4, 3, 2, 3)])
@pytest.mark.parametrize("seed", [1935225697, 1547, 66987, 55571, 998618750])
def test_decompose_and_solve_block_qubo(seed: int, dims: Tuple[int]) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(dims) == 1:
        # Symmetric qubo to handle the case with several solutions
        Q1 = torch.tensor(
            [
                [-1, 2, 2],
                [2, -1, 2],
                [2, 2, -1],
            ],
            dtype=torch.float32,
        )
        Q2 = QUBODataset.from_random(n_matrices=1, matrix_dim=dims[0], densities=[1.0])[0][0]
        blocks = [Q1, Q2]
        N = Q1.shape[0] + dims[0]
    else:
        N = np.sum(dims)
        blocks = [
            QUBODataset.from_random(n_matrices=1, matrix_dim=n, densities=[1.0])[0][0] for n in dims
        ]
    Q = torch.block_diag(*blocks)
    check.equal(Q.shape, (N, N))
    print(f"Qubo matrix:\n{Q}")

    subpb_optimal_bitstrings = []
    for q in blocks:
        results = dict()
        for bits in itertools.product([0, 1], repeat=q.shape[0]):
            z = torch.tensor(bits, dtype=torch.float32)
            cost = (z @ q @ z).item()
            results["".join(str(int(b)) for b in z.flatten())] = cost
        min_cost = min(c for c in results.values())
        subpb_optimal_bitstrings.append(
            {b: c for b, c in results.items() if np.allclose(c, min_cost)}
        )
    print(f"Sub-problems optimal bitstrings: {subpb_optimal_bitstrings}")

    subpb_optimal_bitstrings_list = [list(d.items()) for d in subpb_optimal_bitstrings]
    optimal_bitstrings = {
        "".join(b for b, _ in sub_results): sum(c for _, c in sub_results)
        for sub_results in itertools.product(*subpb_optimal_bitstrings_list)
    }

    print(f"Global optimal bitstrings: {optimal_bitstrings}")

    qubo_instance = QUBOInstance(Q)

    config = SolverConfig(use_quantum=False, decompose=DecompositionConfig(decompose_stop_number=2))
    solver = QuboSolver(qubo_instance, config)
    assert isinstance(solver._solver, DecomposeQuboSolver)

    solution = solver.solve()
    solution.sort_by_cost()
    print(f"Solution: {solution}")
    best_solution = "".join(str(b) for b in solution.bitstrings[0].tolist())
    min_cost = solution.costs[0].item()

    check.is_in(best_solution, optimal_bitstrings.keys())
    check.almost_equal(min_cost, optimal_bitstrings[best_solution])
