from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch
import itertools
import logging
import re
import numpy as np
import pytest
import pytest_check as check
import random

from qubosolver import (
    Instance,
    Solver,
    embedding,
    solvers,
    tensor,
    vector,
    matrix,
    drive_shaping,
    Solution,
    SingleSolution,
)
from qubosolver.utils import analysis
import qoolqit
from qoolqit import DigitalAnalogDevice, AnalogDevice

def gather_optimal_solutions(
    solution: Solution,
) -> list[SingleSolution]:
    min_cost = solution[0].cost
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
    register = qoolqit.Register.from_coordinates(vertices)
    diagonal = (
        torch.ones(4, dtype=vector.dtype())
        if constant_diagonal
        else vector.tensor([1.0, 1.25, 0.2, 1.167])
    )
    Q = matrix.tensor(register.interaction_matrix()) + diagonal_scale * torch.diag(diagonal)
    Q /= Q.max()
    instance = Instance(matrix=Q)

    # Get all bitstrings with minimum cost
    bf_solution = solvers.brute_force(instance, max_bitstrings=10)
    expected_optimal_solutions = gather_optimal_solutions(bf_solution)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.string for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")


    embed_cfg = embedding.Config(
        algorithm="greedy",
        greedy_traps=100,
        greedy_max_possible_term=1.0,
    )

    drive_cfg = drive_shaping.Config(
        algorithm="proportional_diagonal",
        dmm=dmm,
        proportional_diagonal_kappa=0.5,
    )

    config = solvers.Config(
        solving=solvers.QuantumConfig(
            embedding=embed_cfg,
            drive_shaping=drive_cfg,
            device=device_type(),
        )
    )

    solver = Solver(instance, config)
    quantum_solution = solver.solve()

    print(f"{analysis.to_dataframe([quantum_solution])}")

    register = solver._embedding()
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    optimal_solutions = gather_optimal_solutions(quantum_solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.bitstring for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")

    if not constant_diagonal and not dmm:
        pytest.skip("DMM is required to solve Qubos with variable diagonal coefficients")
    if not constant_diagonal and device_type == AnalogDevice:
        pytest.skip(
            "AnalogDevice has no DMM, and cannot solve Qubos with variable diagonal coefficients"
        )

    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)
    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.string, expected_optimal_bitstrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    check.greater(cumulated_probability, 0.75)


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

    with caplog.at_level(logging.INFO, logger="qubosolver.drive_shaping.proportional_diagonal"):
        _ = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device)

    # Since the register cannot be rescaled, limits are the one of the device
    max_amplitude = specs["max_amplitude"]
    max_detuning = specs["max_abs_detuning"]
    assert max_amplitude is not None
    assert max_detuning is not None

    amplitude_match = re.search(
        r"The proportional-diagonal drive amplitude \(([^)]+)\) exceeds the maximum amplitude "
        r"compilable on the device for this register \(([^)]+)\); clamping to it\.",
        caplog.text,
    )
    assert amplitude_match is not None
    check.almost_equal(float(amplitude_match.group(2)), max_amplitude, rel=1e-2)

    detuning_match = re.search(
        r"The proportional-diagonal detuning \(([^)]+)\) exceeds the maximum detuning "
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
    drive = drive_shaping.proportional_diagonal.build_drive(instance, register, dmm=True, device=device)

    assert drive.dmm is not None
    for k, v in drive.dmm.weights.items():
        check.is_instance(k, int)
        check.is_instance(v, float)
    # check that compilation doesn't throw
    qoolqit.QuantumProgram(register, drive).compile_to(device, profile="max_energy", device_max_duration_ratio=0.999)
