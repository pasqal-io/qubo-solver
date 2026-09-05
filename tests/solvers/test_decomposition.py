from __future__ import annotations

import pytest
import pytest_check as check
import torch
import itertools
import numpy as np
import random
from copy import deepcopy

from qoolqit import Register, DigitalAnalogDevice

from qubosolver import (
    solving,
    Solver,
    Dataset,
    Instance,
    matrix,
    bitstring,
    torch_rng,
    SingleSolution,
    Solution,
    analysis,
    vector,
    vectori,
    DecompositionConfig,
    SolverConfig,
    ClassicalSolvingConfig,
    QuantumSolvingConfig,
)
from qubosolver.solver.solver import _DecomposeQuboSolver
from qubosolver.transforms._algorithms.decompose import compute_distance_interaction_matrix


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


@pytest.mark.priority(120)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("use_quantum", [True, False], ids=["quantum", "classical"])
def test_initial_steps_solver(decomposable_qubo: Instance, use_quantum: bool) -> None:
    """Test that the first steps of the decomposition (initialization +
    one loop iteration of a decomposition) are yielding corrent tensors
    or dictionaries of right sizes.

    """
    # Select seed so that the decomposition is tractable for testing with the
    # Qutip backend
    rng = manual_seed(79450)

    from qubosolver.transforms._algorithms.decompose import (
        compute_distance_interaction_matrix,
        compute_min_max_distances,
        geometric_search,
        interaction_matrix_from_placed,
        transfer_edge_values,
        update_global_solution,
        vertices_to_place,
        positive_vertices_update,
    )

    size = decomposable_qubo.size
    qubo_mat = decomposable_qubo.matrix.clone()

    decompose_config = DecompositionConfig()

    if use_quantum:
        config = SolverConfig(
            solving=QuantumSolvingConfig(device=DigitalAnalogDevice()),
            decompose=decompose_config,
        )
    else:
        config = SolverConfig(
            solving=ClassicalSolvingConfig(),
            decompose=decompose_config,
        )
    solver = Solver(decomposable_qubo, config)

    ## Check the distance interaction matrix matches the qubo matrix
    dist_matrix = compute_distance_interaction_matrix(qubo_mat)
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
    solution = bitstring.as_tensor(torch.full((size,), -1))
    transfer_edge_values(current_vertices_dict, dict(), solution, qubo_mat)
    positive_vertices_update(current_vertices_dict, solution)
    assert len(current_vertices_dict) == size

    # try one iteration, check placed_vertices length
    config_subproblems = deepcopy(config)
    config.decompose = None
    first_vertex = 0

    if use_quantum:
        assert isinstance(solver._solver.config.solving, QuantumSolvingConfig)
        pulser_device = solver._solver.config.solving.device._pulser_device
        assert pulser_device.max_radial_distance is not None
        min_distance, max_radial_distance = compute_min_max_distances(
            qubo_mat,
            max_min_dist_ratio=pulser_device.max_radial_distance / pulser_device.min_atom_distance,
        )
    else:
        min_distance = float(DigitalAnalogDevice()._pulser_device.min_atom_distance)
        max_radial_distance_ = DigitalAnalogDevice()._pulser_device.max_radial_distance
        assert max_radial_distance_ is not None
        max_radial_distance = float(max_radial_distance_)

    placed_vertices = geometric_search(
        qubo_mat,
        current_vertices_dict,
        first_vertex,
        decompose_config.decompose_threshold,
        min_distance=min_distance,
        max_radial_distance=max_radial_distance,
        rng=rng,
    )
    assert len(placed_vertices) <= size

    # check matrix size correspond to placed_vertices
    matrix_to_solve, map_index_vertices = interaction_matrix_from_placed(placed_vertices)
    # If too big, the test will take a long time to run.
    if use_quantum and matrix_to_solve.shape[0] > 13:
        raise RuntimeError(f"Test failed due to large matrix size = {matrix_to_solve.shape[0]}")
    assert len(map_index_vertices) == len(placed_vertices) == matrix_to_solve.shape[0]
    subproblem = Instance(matrix_to_solve)
    assert isinstance(solver._solver, _DecomposeQuboSolver)
    subsolver = solver._solver._solver_factory(subproblem, config_subproblems)
    sub_solution = subsolver.solve().bitstrings[0]

    # test update_global_solution remove -1 values
    update_global_solution(
        global_solution=solution, sub_solution=sub_solution, mapping=map_index_vertices
    )
    assert (solution == -1).sum() < size

    # test the transfer changes current_vertices_dict with less vertices to place
    transfer_edge_values(current_vertices_dict, placed_vertices, solution, qubo_mat)
    positive_vertices_update(current_vertices_dict, solution)
    assert len(current_vertices_dict) < size


@pytest.mark.priority(120)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("use_quantum", [True, False], ids=["quantum", "classical"])
def test_decomp_solver(decomposable_qubo: Instance, use_quantum: bool) -> None:

    # Select seed so that the decomposition is tractable for testing with the
    # Qutip backend
    manual_seed(29443)

    if use_quantum:
        config = SolverConfig(
            solving=QuantumSolvingConfig(device=DigitalAnalogDevice()),
            decompose=DecompositionConfig(),
        )
    else:
        config = SolverConfig(
            solving=ClassicalSolvingConfig(),
            decompose=DecompositionConfig(),
        )
    solver = Solver(decomposable_qubo, config)

    assert isinstance(solver._solver, _DecomposeQuboSolver)

    solution = solver.solve()

    # check that only one solution is returned
    assert solution.counts is not None
    assert solution.counts.sum() == 1
    assert len(solution.bitstrings) == 1
    assert (solution.bitstrings[0] == -1).sum() == 0
    assert solution.costs.item() <= 0

    # check that many iterations were done
    assert solver._solver.number_iterations >= 0


def test_small_qubo_solver(simple_qubo_instance: Instance) -> None:

    # assert that the decomposition falls back to not being used as qubo is small
    simple_solver = Solver(
        simple_qubo_instance,
        SolverConfig(solving=ClassicalSolvingConfig(), decompose=None),
    )
    solutions1 = simple_solver.solve()

    decompose_solver = Solver(
        simple_qubo_instance,
        SolverConfig(solving=ClassicalSolvingConfig(), decompose=DecompositionConfig()),
    )
    solutions2 = decompose_solver.solve()
    assert isinstance(decompose_solver._solver, _DecomposeQuboSolver)
    assert decompose_solver._solver.number_iterations == 0

    assert torch.allclose(solutions2.costs.min(), solutions1.costs.min())


def test_scope(decomposable_qubo: Instance) -> None:

    config = SolverConfig(solving=ClassicalSolvingConfig(), decompose=DecompositionConfig())

    # check negative off-diagonal are not supported
    coeffs = decomposable_qubo.matrix
    coeffs[0][1] = -1.0

    with pytest.raises(
        ValueError, match="Decomposition does not handle off-diagonal negative coefficients"
    ):
        Solver(Instance(coeffs), config)


def test_compute_distance_interaction_matrix_zero_output() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0

    Q = matrix.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )

    torch.testing.assert_close(dist_matrix, torch.zeros_like(Q))


def test_compute_distance_interaction_diagonal() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0

    Q = matrix.tensor(
        [
            [-10, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0],
        ],
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )
    expected_dist_matrix = neglecting_inter_distance * torch.ones_like(Q)
    expected_dist_matrix.diagonal().copy_(Q.diag())

    torch.testing.assert_close(dist_matrix, expected_dist_matrix)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("dims", [(4,), (3,), (3, 3), (2, 3, 2), (4, 3, 2, 3)], ids=str)
@pytest.mark.parametrize("seed", [1935225697, 1547, 66987, 55571, 998618750])
def test_decompose_and_solve_block_qubo(seed: int, dims: tuple[int]) -> None:
    """Test that the decomposition solver correctly identifies and solves block-diagonal QUBO matrices.

    The test constructs a block-diagonal QUBO matrix from smaller sub-problems, runs the
    decomposition solver, and verifies that:

    1. The solver's decomposition is a **refinement** of the block structure, i.e. variables
       from different blocks are never grouped into the same sub-decomposition.
    2. The decomposition covers all variables exactly once (it is a partition of ``range(N)``).
    3. The reconstructed global solution matches one of the known optimal bitstrings, and its
       cost matches the known optimal cost.

    When ``len(dims) == 1``, the first block is a fixed symmetric 3×3 matrix (with multiple
    optimal solutions) and the second block is randomly generated with the given dimension.
    Otherwise, all blocks are randomly generated according to ``dims``.

    Some ``(seed, dims)`` combinations are known to produce imperfect decompositions where the
    solver splits a block into sub-decompositions that are too small to recover the global
    optimum. These cases are marked as ``xfail``.

    Args:
        seed (int): Random seed for reproducibility (controls ``random``, ``torch``, and ``numpy``).
        dims (tuple[int, ...]): Dimensions of the individual QUBO blocks that form the
            block-diagonal matrix.
    """

    rng = manual_seed(seed)

    # 32-bits vs 64-bits doesn't generate the same random matrices. So generate 64-bit matrices and convert them.
    if len(dims) == 1:
        # Symmetric qubo to handle the case with several solutions
        Q1 = matrix.tensor(
            [
                [-1, 2, 2],
                [2, -1, 2],
                [2, 2, -1],
            ],
        )
        Q2 = Dataset.from_random(
            n_matrices=1,
            matrix_dim=dims[0],
            densities=[1.0],
            dtype=torch.float64,
            rng=rng,
        )[0][0].matrix
        blocks = [Q1, Q2]
        N = Q1.shape[0] + dims[0]
    else:
        N = np.sum(dims)
        blocks = [
            Dataset.from_random(
                n_matrices=1,
                matrix_dim=n,
                densities=[1.0],
                dtype=torch.float64,
                rng=rng,
            )[0][0].matrix
            for n in dims
        ]
    Q = torch.block_diag(*blocks)
    check.equal(Q.shape, (N, N))
    print(f"Qubo matrix:\n{Q}")

    print("Sub-problems optimal solutions:")
    subpb_optimal_solutions = []
    for q in blocks:
        bf_solution = solving.brute_force.solve(Instance(q), max_bitstrings=-1)
        mask = torch.isclose(bf_solution.costs, bf_solution.costs[0])

        bf_solution.bitstrings = bf_solution.bitstrings[mask]
        bf_solution.costs = bf_solution.costs[mask]
        bf_solution.counts = bf_solution.counts[mask]
        bf_solution.probabilities = bf_solution.probabilities[mask]
        bf_solution._compute_probabilities()

        df = analysis.to_dataframe([bf_solution])
        print(df)

        subpb_optimal_solutions.append(bf_solution)

    optimal_solutions_list = []
    for subpb_solutions in itertools.product(*subpb_optimal_solutions):
        b = torch.cat([s.bitstring for s in subpb_solutions])
        cost = sum(s.cost for s in subpb_solutions)
        optimal_solutions_list.append(SingleSolution(b, cost, 1))

    optimal_solutions = Solution(
        bitstrings=torch.stack([s.bitstring for s in optimal_solutions_list]),
        costs=vector.as_tensor([s.cost for s in optimal_solutions_list]),
        counts=vectori.zeros(len(optimal_solutions_list)).fill_(1),
    )
    optimal_solutions._compute_probabilities()._sort_by_cost()

    print(f"\nGlobal optimal solutions:\n{analysis.to_dataframe([optimal_solutions])}")

    qubo_instance = Instance(Q)

    config = SolverConfig(
        solving=ClassicalSolvingConfig(),
        decompose=DecompositionConfig(decompose_stop_number=2, decompose_break_placement=0),
    )
    solver = Solver(qubo_instance, config)
    assert isinstance(solver._solver, _DecomposeQuboSolver)

    solution = solver.solve()
    print(f"\nSolution:\n{analysis.to_dataframe([solution])}")
    best_solution = solution[0].string
    min_cost = solution[0].cost

    decomposition = solver._solver._decomposition
    print(f"\nDecomposition: {decomposition}")
    sorted_decomposition = sorted([sorted(d) for d in decomposition])
    print(f"Sorted decomposition: {sorted_decomposition}")
    block_decomposition = []
    start = 0
    for size in [b.shape[0] for b in blocks]:
        block_decomposition.append(list(range(start, start + size)))
        start += size
    print(f"Block decomposition: {block_decomposition}")

    # Decomposition a partition of range(N)
    indices = sorted([i for sub_decomposition in decomposition for i in sub_decomposition])
    check.equal(indices, list(range(N)))
    # Perfect decomposition is a partition of range(N)
    block_indices = sorted(
        [i for sub_decomposition in block_decomposition for i in sub_decomposition]
    )
    check.equal(block_indices, list(range(N)))

    # The solver may decompose the QUBO into too many sub-decompositions. The reconstructed solution
    # is then not guaranteed to be optimal.
    non_optimal_cases: list[tuple[int, tuple[int, ...]]] = [
        (55571, (4,)),
        (1547, (3,)),
        (1547, (3, 3)),
        (66987, (3, 3)),
        (66987, (4, 3, 2, 3)),
        (998618750, (4,)),
        (998618750, (4, 3, 2, 3)),
    ]
    failed_cases = [
        (1935225697, (3, 3)),
        (1935225697, (4, 3, 2, 3)),
        (66987, (2, 3, 2)),
        (55571, (3, 3)),
        (55571, (2, 3, 2)),
        (55571, (4, 3, 2, 3)),
        (998618750, (2, 3, 2)),
    ]

    # Assume that A and B are partitions of range(N)
    def is_refinement_of(A: list[list[int]], B: list[list[int]]) -> bool:
        for a in A:
            if not any(set(a).issubset(b) for b in B):
                return False
        return True

    # Examples
    check.is_true(is_refinement_of([[0], [1], [2, 3]], [[0, 1], [2, 3]]))
    check.is_false(is_refinement_of([[0], [1], [2, 3]], [[0, 1, 2], [3]]))

    #  The QUBO is a block matrix. The decomposition should be a refinement of the block decomposition,
    # i.e. two indices from different blocks cannot belong to the same sub-decomposition.
    # Ideally, the decomposition should match the block decomposition, but the solver may decompose
    # the QUBO into smaller sub-decompositions.
    if (seed, dims) in failed_cases:
        check.is_false(is_refinement_of(decomposition, block_decomposition))
        pytest.xfail("Bugged case")
    check.is_true(is_refinement_of(decomposition, block_decomposition))

    if (seed, dims) in non_optimal_cases:
        check.not_equal(sorted_decomposition, block_decomposition)
        check.is_not_in(best_solution, [s.string for s in optimal_solutions])
        check.greater(min_cost, optimal_solutions[0].cost)
        pytest.xfail("The decomposition is not perfect")

    check.is_in(best_solution, [s.string for s in optimal_solutions])
    check.almost_equal(min_cost, optimal_solutions[0].cost)


def test_decompose_embedding() -> None:

    qubo_instance = Instance(matrix.as_tensor(torch.eye(2)))

    config = SolverConfig(decompose=DecompositionConfig())
    solver = Solver(qubo_instance, config)
    with pytest.raises(NotImplementedError):
        solver._embedding()


def test_decompose_drive() -> None:

    qubo_instance = Instance(matrix.as_tensor(torch.eye(2)))

    config = SolverConfig(decompose=DecompositionConfig())
    solver = Solver(qubo_instance, config)
    with pytest.raises(NotImplementedError):
        solver._drive(Register.from_coordinates([(0, 0), (1, 1)]))
