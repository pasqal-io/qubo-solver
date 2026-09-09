from __future__ import annotations

import torch
import numpy as np
import math
import pytest
import pytest_check as check
from unittest.mock import MagicMock, patch
from typing import Iterable

import qoolqit
from qoolqit.devices.device import AnalogDeviceWithDMM, AnalogDevice
from qoolqit.register import Register

from qubosolver import (
    Instance,
    SingleSolution,
    Solution,
    solving,
    vector,
    matrix,
    tensor,
    Tensor,
    Matrix,
    LocalEmulator,
)
from qubosolver.utils import analysis


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
    register = Register.from_coordinates(vertices)
    # Choose scaling factor so that coefficients and costs are in a human readable range (~10)
    Q = 10.0 * matrix.as_tensor(register.interaction_matrix())
    # Choose diagonal coefficients so that the solutions are 011, 101 and 110
    Q = Q - 2.5 * matrix.as_tensor(torch.eye(3)) * Q[0, 1]

    instance = Instance(Q)

    bf_solution = solving.brute_force.solve(instance, max_bitstrings=-1)
    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(bf_solution)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    config = solving.drive_bayesian_search.Config(
        n_evaluations=11,
        seed=seed,
    )
    if use_probability_based_objective:
        config.objective_fn = probability_based_ojective

    qubo_solution, _ = solving.drive_bayesian_search.solve(
        instance,
        register,
        backend=LocalEmulator(),
        device=AnalogDevice(),
        dmm=False,
        config=config,
    )
    print(f"{analysis.to_dataframe([qubo_solution])}")

    check.is_true(qubo_solution.check_consistency(instance=instance, throw=True))
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
    register = Register.from_coordinates(vertices)

    # Choose scaling factor so that coefficients and costs are in a human readable range (~10)
    Q = 400.0 * matrix.as_tensor(register.interaction_matrix())
    # Choose diagonal coefficients so that the solution is 110
    Q = Q - 2.5 * matrix.as_tensor(torch.eye(3)) * Q[0, 1]

    instance = Instance(Q)

    bf_solution = solving.brute_force.solve(instance, max_bitstrings=-1)

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(bf_solution)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    config = solving.drive_bayesian_search.Config(
        n_evaluations=20,
        seed=seed,
    )
    if use_probability_based_objective:
        config.objective_fn = probability_based_ojective

    qubo_solution, _ = solving.drive_bayesian_search.solve(
        instance,
        register,
        backend=LocalEmulator(num_shots=500),
        device=AnalogDevice(),
        dmm=False,
        config=config,
    )
    print(f"{analysis.to_dataframe([qubo_solution])}")

    check.is_true(qubo_solution.check_consistency(instance=instance, throw=True))
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
    register = Register.from_coordinates(vertices)
    Q = matrix.as_tensor(register.interaction_matrix()) - matrix.as_tensor(torch.eye(3))
    instance = Instance(Q)

    def error(solution: Solution) -> float:
        if raise_exception:
            raise RuntimeError("Error occurred")
        return float("inf")

    mock_error = MagicMock(wraps=error)

    config = solving.drive_bayesian_search.Config(
        n_evaluations=11,
        objective_fn=mock_error,
    )

    _, _ = solving.drive_bayesian_search.solve(
        instance,
        register,
        backend=LocalEmulator(),
        device=AnalogDeviceWithDMM(),
        config=config,
    )
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

    seen: list[solving.drive_bayesian_search._CallbackInfo] = []

    def callback(info: solving.drive_bayesian_search._CallbackInfo) -> None:
        seen.append(info)

    bs_config = solving.drive_bayesian_search.Config(n_evaluations=11)
    bs_config._callback_fn = callback

    solving.drive_bayesian_search.solve(
        Instance(Q),
        register,
        backend=backend,
        device=device,
        config=bs_config,
    )

    check.equal(len(seen), 11)
    check.is_instance(seen[0], solving.drive_bayesian_search._CallbackInfo)


def test_failed_simulation() -> None:

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

    config = solving.drive_bayesian_search.Config(
        n_evaluations=11,
    )

    with patch("qoolqit.QuantumProgram.compile_to", side_effect=RuntimeError()):
        _, _ = solving.drive_bayesian_search.solve(
            Instance(Q),
            register,
            backend=LocalEmulator(),
            device=AnalogDeviceWithDMM(),
            config=config,
        )


def test_failed_simulation_2() -> None:

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

    config = solving.drive_bayesian_search.Config(
        n_evaluations=11,
    )
    with patch(
        "qubosolver.solving.drive_bayesian_search._run_simulation",
        return_value=Solution(),
    ):
        solution, _ = solving.drive_bayesian_search.solve(
            Instance(Q),
            register,
            backend=LocalEmulator(),
            device=AnalogDeviceWithDMM(),
            config=config,
        )
        check.is_false(solution)


def test_failed_skopt() -> None:

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

    config = solving.drive_bayesian_search.Config(
        n_evaluations=11,
    )

    with patch("qubosolver.solving.drive_bayesian_search.gp_minimize", return_value=None):
        qubo_solution, _ = solving.drive_bayesian_search.solve(
            Instance(Q),
            register,
            backend=LocalEmulator(),
            device=AnalogDeviceWithDMM(),
            config=config,
        )
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
    _, drive = solving.drive_bayesian_search.solve(
        instance, register, dmm=True, device=device, backend=LocalEmulator()
    )

    assert drive.dmm is not None
    for k, v in drive.dmm.weights.items():
        check.is_instance(k, int)
        check.is_instance(v, float)
    # check that compilation doesn't throw
    qoolqit.QuantumProgram(register, drive).compile_to(
        device, profile="max_energy", device_max_duration_ratio=0.999
    )
