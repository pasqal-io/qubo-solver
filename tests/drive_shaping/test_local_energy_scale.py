from __future__ import annotations

import logging
import random
import re

import numpy as np
import pytest
import pytest_check as check
import torch

import qoolqit
from qoolqit import AnalogDevice, DigitalAnalogDevice

from qubosolver import (
    analysis,
    Instance,
    Solution,
    Solver,
    matrix,
    tensor,
    vector,
    SingleSolution,
    solvers,
    drive_shaping,
    embedding,
)


def gather_optimal_solutions(
    solution: Solution, min_cost: float | None = None
) -> list[SingleSolution]:
    """Return all solutions having the minimum cost."""
    if min_cost is None:
        min_cost = min(s.cost for s in solution)
    return [s for s in solution if np.allclose(s.cost, min_cost)]


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [4548, 33671, 195530])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
@pytest.mark.parametrize("device_type", [DigitalAnalogDevice, AnalogDevice])
@pytest.mark.parametrize("constant_diagonal", [True, False], ids=["cst_diag", "var_diag"])
@pytest.mark.parametrize("diagonal_scale", [-0.9, -3.0, -1.5, -6.0])
def test_with_perfect_embedding(
    seed: int,
    dmm: bool,
    device_type: type[DigitalAnalogDevice] | type[AnalogDevice],
    constant_diagonal: bool,
    diagonal_scale: float,
) -> None:
    """Test the heuristic on exactly embeddable QUBOs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sqrt3 = np.sqrt(3.0)

    vertices = tensor.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
        ]
    )
    interaction_matrix = matrix.tensor(
        qoolqit.Register.from_coordinates(vertices).interaction_matrix()
    )

    diagonal = (
        vector.zeros(4).fill_(1.0) if constant_diagonal else vector.tensor([1.0, 1.25, 0.2, 1.167])
    )
    qubo = interaction_matrix + diagonal_scale * torch.diag(diagonal)
    qubo /= qubo.max()
    instance = Instance(matrix=qubo)

    bf_solutions = solvers.brute_force.solve(instance, max_bitstrings=-1)
    expected_optimal_solutions = gather_optimal_solutions(bf_solutions)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected minimum cost: {expected_optimal_solutions[0].cost}")
    expected_bitstrings = [solution.string for solution in expected_optimal_solutions]
    print(f"Expected optimal bitstrings: {expected_bitstrings}")

    embedding_config = embedding.Config(
        algorithm="greedy_layout",
        greedy_layout_traps=100,
        greedy_layout_max_possible_term=1.0,
    )

    drive_shaping_config = drive_shaping.Config(
        algorithm="local_energy_scale",
        dmm=dmm,
        local_energy_scale_kappa=0.25,
    )

    solving_config = solvers.quantum.Config(
        embedding=embedding_config,
        drive_shaping=drive_shaping_config,
        device=device_type(),
    )

    config = solvers.Config(solving=solving_config)

    solver = Solver(instance, config)
    solution = solver.solve()
    df = analysis.to_dataframe([solution])
    print(df)

    register = solver._embedding()
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    sampled_optimal_solutions = gather_optimal_solutions(solution)
    check.is_not(sampled_optimal_solutions, [])

    minimum_sampled_cost = sampled_optimal_solutions[0].cost

    print(f"\nMinimum sampled cost: {minimum_sampled_cost}")
    print(f"Best sampled bitstrings: {[s.string for s in sampled_optimal_solutions]}")

    if not constant_diagonal and not dmm:
        pytest.skip("DMM is required for variable diagonal coefficients.")

    if not constant_diagonal and device_type is AnalogDevice:
        pytest.skip("AnalogDevice has no DMM and cannot encode variable diagonal coefficients.")

    check.almost_equal(minimum_sampled_cost, expected_optimal_solutions[0].cost)
    expected_bitstrings = [s.string for s in expected_optimal_solutions]
    for solution in sampled_optimal_solutions:
        check.is_in(solution.string, expected_bitstrings)

    cumulated_probability = sum(solution.probability for solution in sampled_optimal_solutions)
    check.greater(cumulated_probability, 0.6)


def test_too_high_diagonal(caplog: pytest.LogCaptureFixture) -> None:

    device = qoolqit.AnalogDeviceWithDMM()
    specs = device.specs
    eps = 0.001
    d = specs["min_distance"]
    assert d is not None
    d += eps
    D = specs["max_radial_distance"]
    assert D is not None
    D -= eps

    # Build a register that cannot be scaled
    register = qoolqit.Register.from_coordinates(
        [
            [0.0, 0.0],
            [d, 0.0],
            [D, 0.0],
        ]
    )
    Q = matrix.tensor(register.interaction_matrix()) + vector.zeros(3).fill_(-50.0).diag()
    instance = Instance(Q)

    with caplog.at_level(logging.INFO, logger="qubosolver.drive_shaping.local_energy_scale"):
        _ = drive_shaping.local_energy_scale.build_drive(instance, register, device=device)

    # Since the register cannot be rescaled, limits are the one of the device
    max_amplitude = specs["max_amplitude"]
    max_detuning = specs["max_abs_detuning"]
    assert max_amplitude is not None
    assert max_detuning is not None

    amplitude_match = re.search(
        r"The local-energy-scale drive amplitude \(([^)]+)\) exceeds the maximum amplitude "
        r"compilable on the device for this register \(([^)]+)\); clamping to it\.",
        caplog.text,
    )
    assert amplitude_match is not None
    check.almost_equal(float(amplitude_match.group(2)), max_amplitude, rel=1e-2)

    detuning_match = re.search(
        r"The local-energy-scale detuning \(([^)]+)\) exceeds the maximum detuning "
        r"compilable on the device for this amplitude \(([^)]+)\); "
        r"scaling the detuning down\.",
        caplog.text,
    )
    assert detuning_match is not None
    check.almost_equal(float(detuning_match.group(2)), max_detuning, rel=1e-2)

def test_dmm_labels_are_ints() -> None:

    vertices = tensor.tensor([[0.0, 0.0], [1.0, 0.0]])
    register = qoolqit.Register.from_coordinates(vertices)
    Q = matrix.as_tensor(register.interaction_matrix()) + torch.diag(vector.tensor([-1.0, -2.0]))
    instance = Instance(Q)

    check.is_instance(register.qubits_ids[0], int)

    device = qoolqit.AnalogDeviceWithDMM()
    drive = drive_shaping.local_energy_scale.build_drive(instance, register, dmm=True, device=device)

    assert drive.dmm is not None
    for k, v in drive.dmm.weights.items():
        check.is_instance(k, int)
        check.is_instance(v, float)
    # check that compilation doesn't throw
    qoolqit.QuantumProgram(register, drive).compile_to(device, profile="max_energy", device_max_duration_ratio=0.999)
