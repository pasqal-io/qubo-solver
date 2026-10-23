from __future__ import annotations

import itertools
import random
from copy import deepcopy

import numpy as np
import pytest
import pytest_check as check
import torch
from qoolqit import DigitalAnalogDevice, Register

from qubosolver import (
    Candidate,
    ClassicalSolvingConfig,
    Dataset,
    Instance,
    QuantumSolvingConfig,
    Solution,
    Solver,
    SolverConfig,
    analysis,
    embedding,
    matrix,
    solving,
    torch_rng,
    vector,
    vectori,
)
from qubosolver.solver.config.config import _DecompositionConfig
from qubosolver.solver.solver import _DecomposeQuboSolver
from qubosolver.transforms import variable_fixing


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


@pytest.mark.duration(120)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("use_quantum", [True, False], ids=["quantum", "classical"])
def test_initial_steps_solver(decomposable_qubo: Instance, use_quantum: bool) -> None:
    """Test that the first steps of the decomposition yield correctly-sized results.

    Covers the presolve and one loop iteration of a decomposition (partial
    embedding, subproblem solving, fixing), yielding results of the right sizes.
    """
    # Select seed so that the decomposition is tractable for testing with the
    # Qutip backend
    rng = manual_seed(79450)

    size = decomposable_qubo.size

    if use_quantum:
        config = SolverConfig(
            solving=QuantumSolvingConfig(device=DigitalAnalogDevice()),
            preprocessing=False,
            postprocessing=False,
        )
    else:
        config = SolverConfig(
            solving=ClassicalSolvingConfig(),
            preprocessing=False,
            postprocessing=False,
        )
    decompose_config = _DecompositionConfig()
    solver = _DecomposeQuboSolver(
        decomposable_qubo, solver_config=config, decompose_config=decompose_config
    )
    assert isinstance(solver, _DecomposeQuboSolver)

    # presolve: the embedder needs negative diagonal coefficients
    reduced = variable_fixing.apply_recursively(decomposable_qubo)
    assert reduced.size <= size
    assert (torch.diag(reduced.matrix) < 0).all()

    # try one iteration, check the number of placed variables
    max_min_dist_ratio = (
        config.quantum.max_min_dist_ratio
        if use_quantum
        else decompose_config.classical_max_min_dist_ratio
    )
    geometric_config = decompose_config._to_geometric_config(max_min_dist_ratio=max_min_dist_ratio)
    register, variables = embedding.geometric.embed(reduced, config=geometric_config, rng=rng)
    assert 1 <= len(variables) <= reduced.size
    assert len(register) == len(variables)

    # check the subproblem size corresponds to the placed variables
    subproblem = reduced.subinstance(variables)
    # If too big, the test will take a long time to run.
    if use_quantum and subproblem.size > 13:
        msg = f"Test failed due to large matrix size = {subproblem.size}"
        raise RuntimeError(msg)
    assert subproblem.size == len(variables)
    subsolver = Solver(subproblem, deepcopy(config))
    sub_solution = subsolver.solve()

    # fixing the subproblem leaves fewer variables to place
    reduced_size = reduced.size
    rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
    variable_fixing.apply(reduced, (rule,), inplace=True)
    variable_fixing.apply_recursively(reduced, inplace=True)
    assert reduced.size <= reduced_size - len(variables)
    assert len(reduced.fixed_variables) == size - reduced.size


@pytest.mark.duration(120)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("use_quantum", [True, False], ids=["quantum", "classical"])
def test_decomp_solver(decomposable_qubo: Instance, use_quantum: bool) -> None:

    # Select seed so that the decomposition is tractable for testing with the
    # Qutip backend
    manual_seed(29443)

    if use_quantum:
        config = SolverConfig(
            solving=QuantumSolvingConfig(device=DigitalAnalogDevice()),
            preprocessing=False,
            postprocessing=False,
        )
    else:
        config = SolverConfig(
            solving=ClassicalSolvingConfig(),
            preprocessing=False,
            postprocessing=False,
        )
    solver = _DecomposeQuboSolver(decomposable_qubo, solver_config=config)

    assert isinstance(solver, _DecomposeQuboSolver)

    solution = solver.solve()

    # check that only one solution is returned
    assert solution.counts is not None
    assert solution.counts.sum() == 1
    assert len(solution.bitstrings) == 1
    assert solution.check_consistency(instance=decomposable_qubo, throw=True)
    check.almost_equal(solution[0].cost, decomposable_qubo.cost(solution[0].bitstring), rel=1e-5)
    assert solution.costs.item() <= 0

    # check that at least one subproblem was solved
    assert solver.number_iterations >= 1


@pytest.mark.usefixtures("restore_rng_state")
def test_decomp_solver_raises_on_empty_sub_solution(
    decomposable_qubo: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A sub-solver returning no candidate (e.g. a quantum run with no sample)
    # is a failure, not something to work around.
    manual_seed(29443)
    monkeypatch.setattr(Solver, "solve", lambda _: Solution())
    config = SolverConfig(
        solving=ClassicalSolvingConfig(), preprocessing=False, postprocessing=False
    )
    solver = _DecomposeQuboSolver(decomposable_qubo, solver_config=config)

    with pytest.raises(RuntimeError, match="no candidate"):
        solver.solve()


def test_decomp_solver_when_hansen_fixes_everything() -> None:
    # Uncoupled variables with negative diagonal coefficients: Hansen's rule
    # fixes them all to 1 in the presolve, so no subproblem is solved.
    size = 20
    instance = Instance(matrix.as_tensor(-torch.eye(size)))
    config = SolverConfig(
        solving=ClassicalSolvingConfig(), preprocessing=False, postprocessing=False
    )
    solver = _DecomposeQuboSolver(instance, solver_config=config)
    assert size > solver.decomposition_config.decompose_stop_number

    solution = solver.solve()

    check.equal(solver.number_iterations, 0)
    check.equal(len(solution), 1)
    check.equal(solution[0].string, "1" * size)
    check.almost_equal(solution[0].cost, -float(size))
    check.is_true(solution.check_consistency(instance=instance, throw=True))


def test_small_qubo_solver(simple_qubo_instance: Instance) -> None:

    # assert that the decomposition falls back to not being used as qubo is small
    simple_solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            solving=ClassicalSolvingConfig(),
            preprocessing=False,
            postprocessing=False,
        ),
    )
    solutions1 = simple_solver.solve()

    decompose_solver = _DecomposeQuboSolver(
        simple_qubo_instance,
        solver_config=SolverConfig(
            solving=ClassicalSolvingConfig(),
            preprocessing=False,
            postprocessing=False,
        ),
    )
    solutions2 = decompose_solver.solve()
    assert isinstance(decompose_solver, _DecomposeQuboSolver)
    assert decompose_solver.number_iterations == 0

    assert torch.allclose(solutions2.costs.min(), solutions1.costs.min())


def test_scope(decomposable_qubo: Instance) -> None:

    config = SolverConfig(
        solving=ClassicalSolvingConfig(),
        preprocessing=False,
        postprocessing=False,
    )

    # check negative off-diagonal are not supported
    coeffs = decomposable_qubo.matrix.clone()
    coeffs[0][1] = coeffs[1][0] = -1.0

    with pytest.raises(
        ValueError, match="Decomposition does not handle off-diagonal negative coefficients"
    ):
        _DecomposeQuboSolver(Instance(coeffs), solver_config=config)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("dims", [(4,), (3,), (3, 3), (2, 3, 2), (4, 3, 2, 3)], ids=str)
@pytest.mark.parametrize("seed", [1935225697, 1547, 66987, 55571, 998618750])
def test_decompose_and_solve_block_qubo(seed: int, dims: tuple[int]) -> None:  # noqa: PLR0915
    """Test that the decomposition solver correctly identifies and solves block-diagonal QUBOs.

    The test constructs a block-diagonal QUBO matrix from smaller sub-problems, runs the
    decomposition solver, and verifies that:

    1. The solver's decomposition is a **refinement** of the block structure, i.e. variables
       from different blocks are never grouped into the same sub-decomposition.
    2. The decomposition covers all variables exactly once (it is a partition of ``range(N)``).
    3. The reconstructed global solution matches one of the known optimal bitstrings, and its
       cost matches the known optimal cost.

    When ``len(dims) == 1``, the first block is a fixed symmetric 3x3 matrix (with multiple
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
    # TODO(abussy-pasqal): Redesign this test. `_DecomposeQuboSolver._decomposition` was
    # removed, so the structural checks below (partition of range(N), refinement of the
    # blocks) no longer have groups to check: rewrite it on the functional decomposition loop
    # (`embedding.geometric` + `variable_fixing`), recording each subproblem's variables and
    # splitting Hansen fixings into singletons.
    pytest.xfail("Flaky tests. Need to be re-designed")

    rng = manual_seed(seed)

    # 32-bits vs 64-bits doesn't generate the same random matrices. So generate 64-bit
    # matrices and convert them.
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
            dtype=matrix.dtype(),
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
                dtype=matrix.dtype(),
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
        optimal_solutions_list.append(Candidate(b, cost, 1))

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
        preprocessing=False,
        postprocessing=False,
    )
    decompose_config = _DecompositionConfig(decompose_stop_number=2, decompose_break_placement=0)
    solver = _DecomposeQuboSolver(
        qubo_instance, solver_config=config, decompose_config=decompose_config
    )

    solution = solver.solve()
    print(f"\nSolution:\n{analysis.to_dataframe([solution])}")
    best_solution = solution[0].string
    min_cost = solution[0].cost

    decomposition = solver._decomposition
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
        (1935225697, (4, 3, 2, 3)),
        (1547, (4,)),
    ]
    failed_cases: list[tuple[int, tuple[int, ...]]] = []

    # Assume that A and B are partitions of range(N)
    def is_refinement_of(A: list[list[int]], B: list[list[int]]) -> bool:
        return all(any(set(a).issubset(b) for b in B) for a in A)

    # Examples
    check.is_true(is_refinement_of([[0], [1], [2, 3]], [[0, 1], [2, 3]]))
    check.is_false(is_refinement_of([[0], [1], [2, 3]], [[0, 1, 2], [3]]))

    #  The QUBO is a block matrix. The decomposition should be a refinement of the block
    # decomposition, i.e. two indices from different blocks cannot belong to the same
    # sub-decomposition. Ideally, the decomposition should match the block decomposition,
    # but the solver may decompose the QUBO into smaller sub-decompositions.
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

    config = SolverConfig(
        preprocessing=False,
        postprocessing=False,
    )
    solver = _DecomposeQuboSolver(qubo_instance, solver_config=config)
    with pytest.raises(NotImplementedError):
        solver._embedding()


def test_decompose_drive() -> None:

    qubo_instance = Instance(matrix.as_tensor(torch.eye(2)))

    config = SolverConfig(
        preprocessing=False,
        postprocessing=False,
    )
    solver = _DecomposeQuboSolver(qubo_instance, solver_config=config)
    with pytest.raises(NotImplementedError):
        solver._drive(Register.from_coordinates([(0, 0), (1, 1)]))
