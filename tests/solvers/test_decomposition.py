from __future__ import annotations

import pytest
import torch
from qubosolver import QUBOInstance

from qubosolver.config import SolverConfig, DecompositionConfig
from qubosolver.solver import DecomposeQuboSolver, QuboSolver


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
    dist_matrix = compute_distance_interaction_matrix(solver._solver.device._pulser_device, qubo_mat)
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
    subsolver = solver._solver._solver_factory(  # type:ignore[attr-defined]
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
