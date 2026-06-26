from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch
import itertools
import numpy as np
import pytest
import pytest_check as check
import random

from qubosolver import (
    QUBOInstance,
    QuboSolver,
    EmbeddingConfig,
    SolverConfig,
    DriveShapingConfig,
    QUBOAnalyzer,
    tensor,
    vector,
)
from qoolqit import DigitalAnalogDevice, AnalogDevice


@dataclass
class Solution:
    bitstring: str
    cost: float = float("inf")
    probability: float = 0.0


def to_solutions(
    bitstrings: Iterable[str | torch.Tensor],
    costs: Iterable[float] = itertools.repeat(float("inf")),
    probabilities: Iterable[float] = itertools.repeat(0.0),
) -> list[Solution]:
    def to_string(b: str | torch.Tensor) -> str:
        if isinstance(b, torch.Tensor):
            return "".join(str(int(i)) for i in b)
        if isinstance(b, str):
            return b
        raise ValueError()

    return [Solution(to_string(b), c, p) for b, c, p in zip(bitstrings, costs, probabilities)]


def gather_optimal_solutions(
    data: Iterable[Solution], min_cost: float | None = None
) -> list[Solution]:
    if min_cost is None:
        min_cost = min(d.cost for d in data)
    return [d for d in data if np.allclose(d.cost, min_cost)]


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

    def interaction_matrix_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
        U = 1.0 / torch.cdist(vertices, vertices) ** 6
        U.fill_diagonal_(0.0)
        return U

    sqrt3 = np.sqrt(3.0)
    vertices = tensor.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
        ]
    )
    diagonal = (
        torch.ones(4, dtype=vector.dtype())
        if constant_diagonal
        else vector.tensor([1.0, 1.25, 0.2, 1.167])
    )
    Q = interaction_matrix_from_vertices(vertices) + diagonal_scale * torch.diag(diagonal)
    Q /= Q.max()

    results = []
    for bits in itertools.product([0, 1], repeat=4):
        z = tensor.tensor(bits)
        cost = (z @ Q @ z).item()
        results.append(Solution("".join(str(int(b)) for b in z.flatten()), cost))

    # Get all bitstrings with minimum cost
    expected_optimal_solutions = gather_optimal_solutions(results)
    check.is_not(expected_optimal_solutions, [])

    print(f"\nExpected Minimum cost: {expected_optimal_solutions[0].cost}")
    print(f"All expected optimal bitstrings: {[s.bitstring for s in expected_optimal_solutions]}")
    print(f"Number of expected optimal solutions: {len(expected_optimal_solutions)}\n")

    instance = QUBOInstance(matrix=Q)

    embed_cfg = EmbeddingConfig(
        embedding_method="greedy",
        greedy_traps=100,
        greedy_spacing=0.1,
        min_distance=1.0001,
    )

    drive_cfg = DriveShapingConfig(
        drive_shaping_method="heuristic",
        dmm=dmm,
        heuristic_kappa=0.5,
    )

    config = SolverConfig(
        use_quantum=True,
        embedding=embed_cfg,
        drive_shaping=drive_cfg,
        device=device_type(),
    )

    solver = QuboSolver(instance, config)
    qubo_solution = solver.solve()

    qubo_solution.sort_by_cost()
    analyzer = QUBOAnalyzer([qubo_solution])
    print(f"{analyzer.df}")

    register = solver.embedding()
    print(f"Register: {register.qubits}")
    print(f"Distances: {register.distances()}")

    assert isinstance(qubo_solution.probabilities, torch.Tensor)
    optimal_solutions = gather_optimal_solutions(
        to_solutions(qubo_solution.bitstrings, qubo_solution.costs, qubo_solution.probabilities)
    )
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
    expected_optimal_bistrings = [s.bitstring for s in expected_optimal_solutions]
    for solution in optimal_solutions:
        check.is_in(solution.bitstring, expected_optimal_bistrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    check.greater(cumulated_probability, 0.75)
