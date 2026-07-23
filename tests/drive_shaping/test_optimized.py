from __future__ import annotations

import torch
import itertools
import numpy as np
import math
import pytest
import pytest_check as check
from unittest.mock import MagicMock, patch
from typing import Iterable, Any

from qoolqit.devices.device import AnalogDeviceWithDMM, AnalogDevice
from qoolqit.register import Register

from qubosolver.drive_shaping._drive_shaper import OptimizedDriveShaper
from qubosolver import (
    Instance,
    SingleSolution,
    Solution,
    SolverConfig,
    DriveShapingConfig,
    Analyzer,
    matrix,
    tensor,
    bitstring,
    Tensor,
    Matrix,
    LocalEmulator,
)
from qubosolver._utils import costs


def interaction_matrix_from_vertices(vertices: Tensor) -> Matrix:
    U = 1.0 / torch.cdist(vertices, vertices) ** 6
    U.fill_diagonal_(0.0)
    return U


def gather_optimal_solutions(
    data: Iterable[SingleSolution], min_cost: float | None = None
) -> list[SingleSolution]:
    if min_cost is None:
        min_cost = min(d.cost for d in data)
    return [d for d in data if np.allclose(d.cost, min_cost)]


def probability_based_ojective(
    solution: Solution,
) -> float:
    optimal_solutions = gather_optimal_solutions(solution)
    check.is_not(optimal_solutions, [])
    min_cost = optimal_solutions[0].cost
    total_prob = sum(s.probability for s in optimal_solutions)
    # Weight in % of cost
    w = math.copysign(0.5, min_cost)
    weighted_cost = min_cost * (1.0 + w * (1.0 - total_prob))

    optimal_bitstrings = [s.bitstring for s in optimal_solutions]
    print(
        f"Best bitstrings: {optimal_bitstrings}, cost: {min_cost}, total probability: {total_prob}, weighted cost: {weighted_cost} "
    )

    return weighted_cost


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [44445, 1217, 990])
@pytest.mark.parametrize("use_probability_based_objective", [True, False])
def test_equilateral_triangular_qubo(seed: int, use_probability_based_objective: bool) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)

    spacing = 1.1

    # Set a Register and compute the associated QUBO
    # Equilateral triangle centered on origin
    vertices = spacing * tensor.tensor(
        [
            [0.0, 1.0 / np.sqrt(3.0)],
            [-0.5, -0.5 / np.sqrt(3.0)],
            [0.5, -0.5 / np.sqrt(3.0)],
        ],
    )
    # Choose scaling factor so that coefficients and costs are in a human readable range (~10)
    Q = 10.0 * interaction_matrix_from_vertices(vertices)
    # Choose diagonal coefficients so that the solutions are 011, 101 and 110
    Q = Q - 2.5 * matrix.from_torch(torch.eye(3)) * Q[0, 1]

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = bitstring.tensor(bits)
        cost = costs.quadratic_cost(z, Q)
        results.append(SingleSolution(z, cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig(drive_shaping_method="optimized")
    if use_probability_based_objective:
        ds_config.optimized_custom_objective = probability_based_ojective
    ds_config.optimized_n_calls = 11
    ds_config.optimized_seed = seed
    config = SolverConfig(device=AnalogDevice(), drive_shaping=ds_config)

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)
    qubo_solution.sort_by_cost()
    analyzer = Analyzer([qubo_solution])
    print(f"{analyzer.df}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(qubo_solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bitstrings = [s.bitstring for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.bitstring, expected_optimal_bitstrings)

    if use_probability_based_objective:
        total_optimal_probability = sum(s.probability for s in optimal_solutions)
        check.greater(total_optimal_probability, 0.75)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.bitstring for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [412, 6983, 5674])
@pytest.mark.parametrize("use_probability_based_objective", [True, False])
def test_triangular_qubo(seed: int, use_probability_based_objective: bool) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)

    spacing = 1.5

    # Set a Register and compute the associated QUBO
    vertices = spacing * tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    # Choose scaling factor so that coefficients and costs are in a human readable range (~10)
    Q = 400.0 * interaction_matrix_from_vertices(vertices)
    # Choose diagonal coefficients so that the solution is 110
    Q = Q - 2.5 * matrix.from_torch(torch.eye(3)) * Q[0, 1]

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = bitstring.tensor(bits)
        cost = costs.quadratic_cost(z, Q)
        results.append(SingleSolution(z, cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig(drive_shaping_method="optimized")
    if use_probability_based_objective:
        ds_config.optimized_custom_objective = probability_based_ojective
    ds_config.optimized_n_calls = 20
    ds_config.optimized_seed = seed
    config = SolverConfig(
        device=AnalogDevice(), drive_shaping=ds_config, backend=LocalEmulator(num_shots=500)
    )

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)
    qubo_solution.sort_by_cost()
    analyzer = Analyzer([qubo_solution])
    print(f"{analyzer.df}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(qubo_solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bitstrings = [s.bitstring for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.bitstring, expected_optimal_bitstrings)

    if use_probability_based_objective:
        total_optimal_probability = sum(s.probability for s in optimal_solutions)
        check.greater(total_optimal_probability, 0.6)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.bitstring for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")


@pytest.mark.parametrize("raise_exception", [True, False])
def test_errors(raise_exception: bool) -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.from_torch(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    def error(solution: Solution) -> float:
        if raise_exception:
            raise RuntimeError("Error occurred")
        return float("inf")

    def optimized_callback_objective(d: dict[Any, Any]) -> None:
        check.almost_equal(d["cost_eval"], 1e4)

    mock_error = MagicMock(wraps=error)
    mock_callback = MagicMock(wraps=optimized_callback_objective)

    ds_config = DriveShapingConfig()
    ds_config.optimized_custom_objective = mock_error
    ds_config.optimized_callback_objective = mock_callback
    ds_config.optimized_n_calls = 11
    config = SolverConfig(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)

    check.equal(mock_error.call_count, 11)
    check.equal(mock_callback.call_count, 11)


def test_failed_simulation() -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.from_torch(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig()
    ds_config.optimized_n_calls = 11
    config = SolverConfig(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)
    with patch("qoolqit.QuantumProgram.compile_to", side_effect=RuntimeError()):
        drive, qubo_solution = drive_shaper.generate(register)


def test_failed_simulation_2() -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.from_torch(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig()
    ds_config.optimized_n_calls = 11
    config = SolverConfig(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)
    with patch(
        "qubosolver.drive_shaping.optimized._run_simulation",
        return_value=Solution(),
    ):
        drive, qubo_solution = drive_shaper.generate(register)
        check.is_true(qubo_solution.empty())


def test_failed_skopt() -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.from_torch(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig()
    ds_config.optimized_n_calls = 11
    config = SolverConfig(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = OptimizedDriveShaper(Instance(Q), config, config.backend)

    with patch("qubosolver.drive_shaping.optimized.gp_minimize", return_value=None):
        drive, qubo_solution = drive_shaper.generate(register)
        check.is_true(qubo_solution.empty())
