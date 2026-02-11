from __future__ import annotations

import pytest
import numpy as np
import torch
import itertools
from typing import Any, Iterable

from qubosolver.solver import QUBOInstance, QuboSolver, QUBOSolution
from qubosolver.config import (
    SolverConfig,
    ClassicalConfig,
    ClassicalSolverType,
    DriveShapingConfig,
    EmbeddingConfig,
)
from qubosolver.qubo_types import DriveType, EmbedderType


def qubo_matrix() -> np.typing.NDArray[np.float32]:
    Q = np.array(
        [
            [-27.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0],
            [4.0, -26.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0],
            [4.0, 3.0, -26.0, 4.0, 3.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, -27.0, 3.0, 4.0, 4.0, 4.0],
            [3.0, 4.0, 3.0, 3.0, -24.0, 4.0, 4.0, 3.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, -26.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 3.0, -26.0, 3.0],
            [4.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, -24.0],
        ],
        dtype=np.float32,
    )
    return Q


def qubo_solution() -> tuple[torch.Tensor, float]:
    return torch.tensor([0, 1, 0, 0, 0, 1, 1, 1], dtype=torch.int32), -62.0


def assert_equivalent_bitstrings(actual: torch.Tensor, expected: torch.Tensor) -> None:
    msg = f"Bitstrings are not equal (up to a flip): actual {actual.tolist()}, expected {expected.tolist()}"
    if actual[0] != expected[0]:
        actual = 1 - actual
    torch.testing.assert_close(actual, expected, msg=msg)


def edges() -> set[tuple[str, str]]:
    return {
        # French side
        ("Paris", "Lyon"),
        ("Lyon", "Marseille"),
        ("Paris", "Marseille"),
        ("Paris", "Calais"),
        # British side
        ("London", "Bristol"),
        ("London", "Edinburgh"),
        ("London", "Dover"),
        # Connections
        ("Calais", "Dover"),
        ("Paris", "London"),
    }


def compare_to_qubovert(
    solutions: QUBOSolution, Q: np.typing.NDArray[np.float32], valid_solutions: list[int] = []
) -> None:  # pragma: no cover

    try:
        import qubovert
    except ImportError:
        return

    def qubovert_reordering(
        problem: qubovert.problems.GraphPartitioning,
    ) -> np.typing.NDArray[np.int64]:
        cities = sorted(list({city for edge in problem.E for city in edge}))
        return np.array([problem._vertex_to_index[c] for c in cities], dtype=np.int64)

    pb = qubovert.problems.GraphPartitioning(edges())
    qubo = pb.to_qubo()
    if () in qubo.keys():
        qubo.pop(())
    qubovert_Q = qubovert.utils.qubo_to_matrix(qubo, symmetric=True)

    reordering = qubovert_reordering(pb)
    inverse_reordering = reordering.copy()
    for i, el in enumerate(reordering):
        inverse_reordering[el] = i

    np.testing.assert_equal(Q, qubovert_Q[np.ix_(reordering, reordering)])

    def readable_solution(bitstring: torch.Tensor) -> Any:
        return pb.convert_solution(bitstring[np.ix_(inverse_reordering)])

    qubovert_solution = qubo.solve_bruteforce()
    qubovert_bitstring = torch.tensor(list(qubovert_solution.values()), dtype=torch.int32)[
        np.ix_(reordering)
    ]
    if qubovert_bitstring[0] != qubo_solution()[0][0]:
        qubovert_bitstring = 1 - qubovert_bitstring
    print()
    print(f"qubovert solution: {readable_solution(qubovert_bitstring)}")

    assert set(valid_solutions) <= set(range(len(solutions.bitstrings)))

    probabilities: Iterable[float] = (
        solutions.probabilities if solutions.probabilities is not None else itertools.repeat(np.nan)
    )
    print()
    for i, (bitstring, cost, probability) in enumerate(
        iterable=zip(solutions.bitstrings, solutions.costs, probabilities)
    ):
        print(
            f"Solution {i} (p = {probability:.4f}): {readable_solution(bitstring)}, cost = {cost}"
        )
        if i in valid_solutions:
            assert pb.is_solution_valid(readable_solution(bitstring))
            assert_equivalent_bitstrings(bitstring, qubovert_bitstring)


@pytest.mark.priority(30)
@pytest.mark.parametrize(
    "classical_method", [c.value for c in ClassicalSolverType if c.value != "random"]
)
def test_compare_classical_to_qubovert(classical_method: str) -> None:
    torch.set_printoptions(profile="full")

    Q = qubo_matrix()
    instance = QUBOInstance(Q)

    config = SolverConfig(classical=ClassicalConfig(classical_solver_type=classical_method))
    config.classical.sa_seed = 48
    solver = QuboSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions.bitstrings) == 1

    print()
    for i, (solution, cost) in enumerate(iterable=zip(solutions.bitstrings, solutions.costs)):
        print(f"Solution {i}: {solution}, cost: {cost}")

    b, c = qubo_solution()
    assert_equivalent_bitstrings(solutions.bitstrings[0], b)
    assert solutions.costs[0] == c

    compare_to_qubovert(solutions, Q, [0])


@pytest.mark.priority(60)
@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("embedding_method", list(EmbedderType))
def test_compare_quantum_to_qubovert(drive_method: str, embedding_method: str) -> None:

    Q = qubo_matrix()
    instance = QUBOInstance(Q)

    config = SolverConfig(use_quantum=True)
    config.drive_shaping = DriveShapingConfig(
        drive_shaping_method=drive_method,
        optimized_n_calls=11,
    )
    config.embedding = EmbeddingConfig(embedding_method=embedding_method)
    solver = QuboSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions.bitstrings) >= 1
    assert solutions.probabilities is not None

    print()
    for i, (bitstring, cost, probability) in enumerate(
        iterable=zip(solutions.bitstrings, solutions.costs, solutions.probabilities)
    ):
        print(f"Solution {i} (p = {probability:.4f}): {bitstring.tolist()}, cost = {cost}")

    # TODO: find a relevent test
    # b, c = qubo_solution()
    # assert_equivalent_bitstrings(solutions.bitstrings[0], b)
    # assert solutions.costs[0] == c

    compare_to_qubovert(solutions, Q, [])
