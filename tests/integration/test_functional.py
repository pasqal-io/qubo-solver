# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check
import qoolqit
import scipy

from qubosolver import (
    Candidate,
    Instance,
    Solution,
    drive_shaping,
    embedding,
    numpy_rng,
    solving,
    transforms,
)
from qubosolver.utils import analysis

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gather_optimal_solutions(solutions: Solution) -> list[Candidate]:
    min_cost = solutions[0].cost
    return [d for d in solutions if np.allclose(d.cost, min_cost)]


def interaction_matrix_from_vertices(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        U = 1.0 / scipy.spatial.distance.cdist(vertices, vertices) ** 6
    np.fill_diagonal(U, 0.0)
    return np.asarray(U, dtype=np.float64)


def simple_qubo() -> tuple[Instance, list[Candidate]]:

    sqrt3 = np.sqrt(3.0)
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )
    print(vertices)
    d = scipy.spatial.distance.cdist(vertices, vertices)
    print(d)
    n_qubits = vertices.shape[0]
    diagonal_scale = -2.0
    diagonal = np.full(n_qubits, 1.0)
    Q = interaction_matrix_from_vertices(vertices) + diagonal_scale * np.diag(diagonal)
    Q /= Q.max()

    solutions = Solution()
    solutions.bitstrings = np.asarray(
        list(itertools.product([0, 1], repeat=n_qubits)), dtype=np.int8
    )
    solutions.counts = np.full(solutions.bitstrings.shape[0], 1, dtype=np.int64)
    solutions._compute_costs(Q)._sort_by_cost()._compute_probabilities()

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(solutions)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    return Instance(matrix=Q), expected_optimal_solutions


def manual_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    random.seed(seed)
    return numpy_rng(seed)


def check_solution(
    solutions: Solution,
    expected_optimal_solutions: list[Candidate],
    *,
    expected_optimal_probability: float = 0.75,
) -> None:

    # Solutions are not duplicated
    check.equal(np.unique(solutions.bitstrings, axis=0).shape[0], solutions.bitstrings.shape[0])

    print(f"{analysis.to_dataframe([solutions])}")

    assert isinstance(solutions.probabilities, np.ndarray)
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


@pytest.mark.duration(7)
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
    trivial_solution = solving.trivial_solution_search.solve(qubo)
    check.is_false(trivial_solution)

    effective_qubo = qubo
    if preprocessing:
        effective_qubo = transforms.variable_fixing.apply_recursively(qubo)

    if embedding_method == "blade":
        register = embedding.blade.embed_for_device(effective_qubo, device)
    elif embedding_method == "greedy_layout":
        greedy_config = embedding.greedy_layout.Config.from_device(device)
        greedy_config.traps = 100
        register = embedding.greedy_layout.embed(effective_qubo, config=greedy_config)
    else:  # no cov
        msg = f"Invalid embedding method: {embedding_method}"
        raise ValueError(msg)
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
        _, drive = solving.drive_bayesian_search.solve(
            effective_qubo,
            register,
            backend=emulator,
            device=device,
            dmm=False,
            config=solving.drive_bayesian_search.Config(n_evaluations=11, seed=seed),
        )
    else:  # no cov
        msg = f"Invalid drive shaping method: {drive_shaping_method}"
        raise ValueError(msg)

    program = solving.analog_quantum_sampling.compile(register, drive, device)
    job = emulator.run(program)
    solution = Solution.from_results(job.results(), effective_qubo)

    # Post-process fixations of the preprocessing and restore the original QUBO
    if preprocessing:
        assert isinstance(effective_qubo, transforms.variable_fixing.Instance)
        solution = transforms.variable_fixing.lift(solution, effective_qubo)

    if postprocessing:
        solution = solving.iterative_bitflip_local_search.solve(qubo, starts=solution)

    expected_optimal_probability = 0.75
    if drive_shaping_method == "bayesian_search":
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "solving_method",
    [pytest.param("cplex", marks=pytest.mark.extras), "tabu", "sa", "sa+tabu", "random"],
)
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
    trivial_solution = solving.trivial_solution_search.solve(qubo)
    check.is_false(trivial_solution)

    effective_qubo = qubo
    if preprocessing:
        effective_qubo = transforms.variable_fixing.apply_recursively(qubo)

    if solving_method == "cplex":
        solution = solving.cplex.solve(effective_qubo)
    elif solving_method == "tabu":
        solution = solving.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=3)
        solution = solving.tabu_search.solve(effective_qubo, starts=solution.bitstrings)
    elif solving_method == "sa":
        solution = solving.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=1)
        solution = solving.simulated_annealing.solve(
            effective_qubo, starts=solution[0].bitstring[np.newaxis], top_k=1
        )
    elif solving_method == "sa+tabu":
        solution = solving.random_sampling.solve(effective_qubo, rng=rng, max_bitstrings=1)
        solution = solving.simulated_annealing.solve(
            effective_qubo, starts=solution[0].bitstring[np.newaxis], top_k=1
        )
        solution = solving.tabu_search.solve(effective_qubo, starts=solution.bitstrings)
    elif solving_method == "random":
        solution = solving.random_sampling.solve(effective_qubo, rng=rng)
    else:  # no cov
        msg = f"Invalid solving method: {solving_method}"
        raise ValueError(msg)

    if preprocessing:
        assert isinstance(effective_qubo, transforms.variable_fixing.Instance)
        solution = transforms.variable_fixing.lift(solution, effective_qubo)

    if postprocessing:
        solution = solving.iterative_bitflip_local_search.solve(qubo, starts=solution)

    expected_optimal_probability = 0.75
    if solving_method == "random":
        expected_optimal_probability = 0.0

    check_solution(
        solution,
        expected_optimal_solutions,
        expected_optimal_probability=expected_optimal_probability,
    )
