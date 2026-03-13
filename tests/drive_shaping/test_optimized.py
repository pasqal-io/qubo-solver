from __future__ import annotations

from dataclasses import dataclass
import torch
import itertools
import numpy as np
import math
import pytest
import pytest_check as check
from typing import List, Iterable

from qoolqit.devices.device import DigitalAnalogDevice
from qoolqit.register import Register
from qoolqit.execution import LocalEmulator

from qubosolver.pipeline.drive import OptimizedDriveShaper
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.config import SolverConfig, DriveShapingConfig
from qubosolver.qubo_analyzer import QUBOAnalyzer


def triangular_qubo() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def interaction_matrix_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
    n = vertices.shape[0]
    U = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            U[i, j] = 1.0 / torch.norm(vertices[i] - vertices[j]) ** 6
            U[j, i] = U[i, j]
    return U


@dataclass
class Solution:
    bitstring: str
    cost: float = float("inf")
    probability: float = 0.0


def to_solutions(
    bitstrings: Iterable[str | torch.Tensor],
    costs: Iterable[float] = itertools.repeat(float("inf")),
    probabilities: Iterable[float] = itertools.repeat(0.0),
) -> List[Solution]:
    def to_string(b: str | torch.Tensor) -> str:
        if isinstance(b, torch.Tensor):
            return "".join(str(int(i)) for i in b)
        if isinstance(b, str):
            return b
        raise ValueError()

    return [Solution(to_string(b), c, p) for b, c, p in zip(bitstrings, costs, probabilities)]


def gather_optimal_solutions(
    data: Iterable[Solution], min_cost: float | None = None
) -> List[Solution]:
    if min_cost is None:
        min_cost = min(d.cost for d in data)
    return [d for d in data if np.allclose(d.cost, min_cost)]


def probability_based_ojective(
    bitstrings: list,
    counts: list,
    probabilities: list,
    costs: list,
    best_cost: float,
    best_bitstring: str,
) -> float:
    optimal_solutions = gather_optimal_solutions(to_solutions(bitstrings, costs, probabilities))
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
@pytest.mark.parametrize("seed", [44445, 1217, 998])
@pytest.mark.parametrize("use_probability_based_ojective", [True, False])
def test_equilateral_triangular_qubo(seed: int, use_probability_based_ojective: bool) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = DigitalAnalogDevice()._device
    C6 = device.interaction_coeff
    spacing = 7.0

    # Set a Register and compute the associated QUBO
    # Equilateral triangle centered on origin
    vertices = spacing * torch.tensor(
        [
            [0.0, 1.0 / np.sqrt(3.0)],
            [-0.5, -0.5 / np.sqrt(3.0)],
            [0.5, -0.5 / np.sqrt(3.0)],
        ],
        dtype=torch.float32,
    )
    Q = C6 * interaction_matrix_from_vertices(vertices) - 100.0 * torch.eye(3, dtype=torch.float32)

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = torch.tensor(bits, dtype=torch.float32)
        cost = (z @ Q @ z).item()
        results.append(Solution("".join(str(int(b)) for b in z.flatten()), cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig()
    if use_probability_based_ojective:
        ds_config.optimized_custom_objective = probability_based_ojective
    ds_config.optimized_n_calls = 11
    config = SolverConfig(device=DigitalAnalogDevice(), drive_shaping=ds_config)

    drive_shaper = OptimizedDriveShaper(QUBOInstance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(
        to_solutions(qubo_solution.bitstrings, qubo_solution.costs, qubo_solution.probabilities)
    )
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bistrings = [s.bitstring for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.bitstring, expected_optimal_bistrings)

    if use_probability_based_ojective:
        total_optimal_probability = sum(s.probability for s in optimal_solutions)
        check.greater(total_optimal_probability, 0.35)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.bitstring for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [412, 6983, 5674])
@pytest.mark.parametrize("use_probability_based_ojective", [True, False])
def test_triangular_qubo(seed: int, use_probability_based_ojective: bool) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)

    spacing = 2.0

    # Set a Register and compute the associated QUBO
    # Equilateral triangle centered on origin
    vertices = spacing * torch.tensor(
        [
            [0.0, 0.5],
            [-0.8, -0.4],
            [0.2, -0.15],
        ],
        dtype=torch.float32,
    )
    Q = 200.0 * (
        interaction_matrix_from_vertices(vertices) - 0.05 * torch.eye(3, dtype=torch.float32)
    )

    results = []
    for bits in itertools.product([0, 1], repeat=3):
        z = torch.tensor(bits, dtype=torch.float32)
        cost = (z @ Q @ z).item()
        results.append(Solution("".join(str(int(b)) for b in z.flatten()), cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    register = Register.from_coordinates(vertices.tolist())

    ds_config = DriveShapingConfig()
    if use_probability_based_ojective:
        ds_config.optimized_custom_objective = probability_based_ojective
    ds_config.optimized_n_calls = 11
    config = SolverConfig(
        device=DigitalAnalogDevice(), drive_shaping=ds_config, backend=LocalEmulator(runs=500)
    )

    drive_shaper = OptimizedDriveShaper(QUBOInstance(Q), config, config.backend)
    drive, qubo_solution = drive_shaper.generate(register)
    qubo_solution.sort_by_cost()
    analyzer = QUBOAnalyzer([qubo_solution])
    print(f"{analyzer.df}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(
        to_solutions(qubo_solution.bitstrings, qubo_solution.costs, qubo_solution.probabilities)
    )
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)

    expected_optimal_bistrings = [s.bitstring for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.bitstring, expected_optimal_bistrings)

    total_optimal_probability = sum(s.probability for s in optimal_solutions)
    check.greater(total_optimal_probability, 0.25)

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.bitstring for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")
