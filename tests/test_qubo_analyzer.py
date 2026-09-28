from __future__ import annotations

import pytest
import torch

from qubosolver.data import QUBOSolution
from qubosolver.qubo_analyzer import QUBOAnalyzer


# @VV: I didn't get, should I define it as a fixture?
@pytest.fixture
def basic_solution() -> QUBOSolution:
    return QUBOSolution(
        bitstrings=torch.tensor([[0, 1, 0], [1, 0, 1]]),
        costs=torch.tensor([1.0, 2.0]),
        counts=torch.tensor([15, 5]),
        probabilities=torch.tensor([0.75, 0.25]),
    )


@pytest.fixture
def analyzer(basic_solution: QUBOSolution) -> QUBOAnalyzer:
    return QUBOAnalyzer(solutions=[basic_solution], labels=["sol1"])


def test_init_single_solution(basic_solution: QUBOSolution) -> None:
    analyzer = QUBOAnalyzer(solutions=basic_solution)
    assert len(analyzer.solutions) == 1
    assert analyzer.labels == ["0"]


def test_tensor_to_bitstrings() -> None:
    tensor = torch.tensor([[1, 0, 1], [0, 1, 0]])
    expected = ["101", "010"]
    assert QUBOAnalyzer.tensor_to_bitstrings(tensor) == expected


def test_bitstrings_to_tensor() -> None:
    strings = ["101", "010"]
    expected = torch.tensor([[1, 0, 1], [0, 1, 0]])
    result = QUBOAnalyzer.bitstrings_to_tensor(strings)
    assert torch.equal(result, expected)


def test_solution_to_dataframe(analyzer: QUBOAnalyzer, basic_solution: QUBOSolution) -> None:
    df = analyzer._solution_to_dataframe(basic_solution, "sol1")
    assert "bitstrings" in df.columns
    assert "probs" in df.columns
    assert "costs" in df.columns
    assert "counts" in df.columns
    assert len(df) == 2


def test_to_dataframe(analyzer: QUBOAnalyzer) -> None:
    df = analyzer._to_dataframe()
    assert "bitstrings" in df.columns
    assert "probs" in df.columns
    assert "costs" in df.columns
    assert "counts" in df.columns


def test_filter_by_probability(analyzer: QUBOAnalyzer) -> None:
    df = analyzer.filter_by_probability(0.5)
    assert all(df["probs"] > 0.5)


def test_filter_by_cost(analyzer: QUBOAnalyzer) -> None:
    df = analyzer.filter_by_cost(1.5)
    assert all(df["costs"] < 1.5)


def test_filter_by_percentage(analyzer: QUBOAnalyzer) -> None:
    df = analyzer.filter_by_percentage(top_percent=0.5)
    assert df["probs"].sum() >= 0.5


def test_average_cost(analyzer: QUBOAnalyzer) -> None:
    result = analyzer.average_cost(top_percent=1.0)
    assert "average cost" in result.columns


def test_best_bitstrings(analyzer: QUBOAnalyzer) -> None:
    result = analyzer.best_bitstrings()
    assert "bitstrings" in result.columns


def test_add_counts(analyzer: QUBOAnalyzer) -> None:
    analyzer.add_counts([15, 5])
    assert "counts" in analyzer.df.columns


def test_add_probs(analyzer: QUBOAnalyzer) -> None:
    analyzer.add_probs([0.75, 0.25])
    assert "probs" in analyzer.df.columns


def test_calculate_gaps(analyzer: QUBOAnalyzer) -> None:
    df = analyzer.calculate_gaps(opt_cost=1.0)
    assert "gaps" in df.columns
