from __future__ import annotations

import torch
import itertools
import numpy as np
import math
import pytest
import pytest_check as check
from unittest.mock import MagicMock, patch
from typing import Iterable

import qoolqit
from qoolqit.devices.device import AnalogDeviceWithDMM, AnalogDevice
from qoolqit.register import Register

from qubosolver.drive_shaping._drive_shaper import BayesianSearchDriveShaper
from qubosolver.solvers.hybrid import drive_bayesian_search
from qubosolver import (
    Instance,
    SingleSolution,
    Solution,
    solvers,
    drive_shaping,
    vector,
    matrix,
    tensor,
    bitstring,
    Tensor,
    Matrix,
    LocalEmulator,
)
from qubosolver.utils import _costs, analysis


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

    optimal_bitstrings = [s.string for s in optimal_solutions]
    print(
        f"Best bitstrings: {optimal_bitstrings}, cost: {min_cost}, total probability: {total_prob}, weighted cost: {weighted_cost} "
    )

    return weighted_cost


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [44445, 1218, 990])
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
    Q = Q - 2.5 * matrix.as_tensor(torch.eye(3)) * Q[0, 1]

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = bitstring.tensor(bits)
        cost = _costs.quadratic_cost(z, Q)
        results.append(SingleSolution(z, cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = drive_shaping.Config(algorithm="bayesian_search")
    if use_probability_based_objective:
        ds_config.bayesian_search_custom_objective = probability_based_ojective
    ds_config.bayesian_search_n_calls = 11
    ds_config.bayesian_search_seed = seed
    config = solvers.quantum.Config(device=AnalogDevice(), drive_shaping=ds_config)

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)
    qubo_solution._sort_by_cost()
    print(f"{analysis.to_dataframe([qubo_solution])}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(qubo_solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.string, expected_optimal_bitstrings)

    if use_probability_based_objective:
        total_optimal_probability = sum(s.probability for s in optimal_solutions)
        check.greater(total_optimal_probability, 0.3)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [600, 6983, 5674])
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
    Q = Q - 2.5 * matrix.as_tensor(torch.eye(3)) * Q[0, 1]

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = bitstring.tensor(bits)
        cost = _costs.quadratic_cost(z, Q)
        results.append(SingleSolution(z, cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = drive_shaping.Config(algorithm="bayesian_search")
    if use_probability_based_objective:
        ds_config.bayesian_search_custom_objective = probability_based_ojective
    ds_config.bayesian_search_n_calls = 20
    ds_config.bayesian_search_seed = seed
    config = solvers.quantum.Config(
        device=AnalogDevice(), drive_shaping=ds_config, backend=LocalEmulator(num_shots=500)
    )

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)
    qubo_solution._sort_by_cost()
    print(f"{analysis.to_dataframe([qubo_solution])}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(qubo_solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.string, expected_optimal_bitstrings)

    if use_probability_based_objective:
        total_optimal_probability = sum(s.probability for s in optimal_solutions)
        check.greater(total_optimal_probability, 0.1)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
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
    Q = interaction_matrix_from_vertices(vertices) - matrix.as_tensor(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    def error(solution: Solution) -> float:
        if raise_exception:
            raise RuntimeError("Error occurred")
        return float("inf")

    mock_error = MagicMock(wraps=error)

    ds_config = drive_shaping.Config()
    ds_config.bayesian_search_custom_objective = mock_error
    ds_config.bayesian_search_n_calls = 11
    config = solvers.quantum.Config(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)

    check.equal(mock_error.call_count, 11)


def test_callback_fn() -> None:
    """`_callback_fn` is private/experimental: set it by attribute after construction."""

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    register = Register.from_coordinates(vertices)
    Q = matrix.as_tensor(register.interaction_matrix()) - matrix.as_tensor(torch.eye(3))

    device = AnalogDeviceWithDMM()
    backend = LocalEmulator()

    seen: list[bayesian_search._CallbackInfo] = []

    def callback(info: bayesian_search._CallbackInfo) -> None:
        seen.append(info)

    bs_config = bayesian_search.Config(n_evaluations=11)
    bs_config._callback_fn = callback

    bayesian_search.build_drive(
        Instance(Q),
        register,
        backend=backend,
        device=device,
        config=bs_config,
    )

    check.equal(len(seen), 11)
    check.is_instance(seen[0], bayesian_search._CallbackInfo)


def test_failed_simulation() -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.as_tensor(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = drive_shaping.Config()
    ds_config.bayesian_search_n_calls = 11
    config = solvers.quantum.Config(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)
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
    Q = interaction_matrix_from_vertices(vertices) - matrix.as_tensor(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = drive_shaping.Config()
    ds_config.bayesian_search_n_calls = 11
    config = solvers.quantum.Config(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)
    with patch(
        "qubosolver.drive_shaping.bayesian_search._run_simulation",
        return_value=Solution(),
    ):
        _, qubo_solution = drive_shaper.generate(register)
        check.is_false(qubo_solution)


def test_failed_skopt() -> None:

    # Set a Register and compute the associated QUBO
    vertices = tensor.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
    )
    Q = interaction_matrix_from_vertices(vertices) - matrix.as_tensor(torch.eye(3))

    register = Register.from_coordinates(vertices.tolist())

    ds_config = drive_shaping.Config()
    ds_config.bayesian_search_n_calls = 11
    config = solvers.quantum.Config(
        device=AnalogDeviceWithDMM(),
        drive_shaping=ds_config,
    )

    drive_shaper = BayesianSearchDriveShaper(Instance(Q), config, config.backend)

    with patch("qubosolver.drive_shaping.bayesian_search.gp_minimize", return_value=None):
        _, qubo_solution = drive_shaper.generate(register)
        # Falls back to the default x0 parameters, which are now clamped to
        # stay compilable on the device, so the simulation succeeds.
        check.is_true(qubo_solution)

def test_dmm_labels_are_ints() -> None:

    vertices = tensor.tensor([[0.0, 0.0], [1.0, 0.0]])
    register = qoolqit.Register.from_coordinates(vertices)
    Q = matrix.as_tensor(register.interaction_matrix()) + torch.diag(vector.tensor([-1.0, -2.0]))
    instance = Instance(Q)

    check.is_instance(register.qubits_ids[0], int)

    device = qoolqit.AnalogDeviceWithDMM()
    drive, _ = drive_shaping.bayesian_search.build_drive(instance, register, dmm=True, device=device, backend=LocalEmulator())

    assert drive.dmm is not None
    for k, v in drive.dmm.weights.items():
        check.is_instance(k, int)
        check.is_instance(v, float)
    # check that compilation doesn't throw
    qoolqit.QuantumProgram(register, drive).compile_to(device, profile="max_energy", device_max_duration_ratio=0.999)
