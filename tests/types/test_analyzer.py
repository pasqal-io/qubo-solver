from __future__ import annotations

import pytest
from typing import get_args

from qubosolver import (
    Instance,
    Solution,
    Solver,
    SolverConfig,
    ClassicalSolvingConfig,
    QuantumSolvingConfig,
)
from qubosolver.solver.config.solving import ClassicalAlgorithm
from qubosolver.utils import analysis


def test_to_dataframe_single_solution(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    check_labels = df["labels"].unique().tolist()
    assert check_labels == ["0"]


def test_to_dataframe_with_labels(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution], labels=["sol1"])
    assert df["labels"].unique().tolist() == ["sol1"]


def test_to_dataframe_errors(basic_solution: Solution) -> None:
    with pytest.raises(ValueError):
        analysis.to_dataframe([basic_solution], labels=["0", "1"])


def test_solution_to_dataframe(basic_solution: Solution) -> None:
    df = analysis._solution_to_dataframe(basic_solution, "sol1")
    assert "bitstrings" in df.columns
    assert "probs" in df.columns
    assert "costs" in df.columns
    assert "counts" in df.columns
    assert len(df) == 2


def test_filter_by_probability(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    filtered = df[df["probs"] > 0.5]
    assert all(filtered["probs"] > 0.5)


def test_filter_by_cost(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    filtered = df[df["costs"] < 1.5]
    assert all(filtered["costs"] < 1.5)


def test_filter_by_percentage(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    filtered = analysis._filter_by_percentage(df, top_percent=0.5, column="probs", order="descending")
    assert filtered["probs"].sum() >= 0.5


def test_average_cost(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    result = analysis._average_cost(df, top_percent=1.0)
    assert "average cost" in result.columns


def test_best_bitstrings(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    result = analysis._best_bitstrings(df)
    assert "bitstrings" in result.columns


def test_calculate_gaps(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    df = analysis._add_gaps(df, opt_cost=1.0)
    assert "gaps" in df.columns


@pytest.mark.parametrize("classical_method", get_args(ClassicalAlgorithm))
def test_analyzer_classical(simple_qubo_instance: Instance, classical_method: ClassicalAlgorithm) -> None:
    config = SolverConfig(
        solving=ClassicalSolvingConfig(algorithm=classical_method),
    )
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()
    df = analysis.to_dataframe([solution], labels=["sol1"])

    assert len(df) == len(solution.bitstrings)
    assert "counts" in df.columns
    assert "probs" in df.columns


def test_analyzer_quantum(simple_qubo_instance: Instance) -> None:
    config = SolverConfig(solving=QuantumSolvingConfig())
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()
    df = analysis.to_dataframe([solution], labels=["sol1"])

    assert len(df) == len(solution.bitstrings)
    assert "probs" in df.columns
    assert "counts" in df.columns


@pytest.mark.parametrize("classical_method", get_args(ClassicalAlgorithm))
def test_analyzer_quantum_and_classical(
    simple_qubo_instance: Instance, classical_method: ClassicalAlgorithm
) -> None:
    config = SolverConfig(
        solving=ClassicalSolvingConfig(algorithm=classical_method),
    )
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()

    quantumsolver = Solver(simple_qubo_instance, SolverConfig(solving=QuantumSolvingConfig()))
    quantumsolution = quantumsolver.solve()
    df = analysis.to_dataframe([solution, quantumsolution], labels=["sol1", "sol2"])

    assert len(df) == len(solution.bitstrings) + len(quantumsolution.bitstrings)
