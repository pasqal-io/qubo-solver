from __future__ import annotations

import pytest

from qubosolver import (
    Analyzer,
    Instance,
    Solution,
    SolverConfig,
    ClassicalConfig,
    ClassicalSolverType,
    Solver,
    vector,
    vectori,
)


def test_init_single_solution(basic_solution: Solution) -> None:
    analyzer = Analyzer(solutions=basic_solution)
    assert len(analyzer.solutions) == 1
    assert analyzer.labels == ["0"]

    analyzer = Analyzer(solutions=basic_solution, labels="0")
    assert len(analyzer.solutions) == 1
    assert analyzer.labels == ["0"]


def test_errors(basic_solution: Solution) -> None:

    with pytest.raises(TypeError):
        Analyzer(solutions=[basic_solution, 0])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        Analyzer(solutions=basic_solution, labels=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Analyzer(solutions=basic_solution, labels=["0", "1"])


def test_solution_to_dataframe(analyzer: Analyzer, basic_solution: Solution) -> None:
    df = analyzer._solution_to_dataframe(basic_solution, "sol1")
    assert "bitstrings" in df.columns
    assert "probs" in df.columns
    assert "costs" in df.columns
    assert "counts" in df.columns
    assert len(df) == 2


def test_to_dataframe(analyzer: Analyzer) -> None:
    df = analyzer._to_dataframe()
    assert "bitstrings" in df.columns
    assert "probs" in df.columns
    assert "costs" in df.columns
    assert "counts" in df.columns


def test_filter_by_probability(analyzer: Analyzer) -> None:
    df = analyzer.filter_by_probability(0.5)
    assert all(df["probs"] > 0.5)


def test_filter_by_cost(analyzer: Analyzer) -> None:
    df = analyzer.filter_by_cost(1.5)
    assert all(df["costs"] < 1.5)


def test_filter_by_percentage(analyzer: Analyzer) -> None:
    df = analyzer.filter_by_percentage(top_percent=0.5)
    assert df["probs"].sum() >= 0.5


def test_average_cost(analyzer: Analyzer) -> None:
    result = analyzer.average_cost(top_percent=1.0)
    assert "average cost" in result.columns


def test_best_bitstrings(analyzer: Analyzer) -> None:
    result = analyzer.best_bitstrings()
    assert "bitstrings" in result.columns


def test_add_counts(analyzer: Analyzer) -> None:
    analyzer.add_counts(vectori.tensor([15, 5]))
    assert "counts" in analyzer.df.columns


def test_add_probs(analyzer: Analyzer) -> None:
    analyzer.add_probs(vector.tensor([0.75, 0.25]))
    assert "probs" in analyzer.df.columns


def test_calculate_gaps(analyzer: Analyzer) -> None:
    df = analyzer.calculate_gaps(opt_cost=1.0)
    assert "gaps" in df.columns


@pytest.mark.parametrize("classical_method", [c.value for c in ClassicalSolverType])
def test_analyzer_classical(simple_qubo_instance: Instance, classical_method: str) -> None:
    config = SolverConfig(
        use_quantum=False, classical=ClassicalConfig(classical_solver_type=classical_method)
    )
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()
    analyzer = Analyzer([solution], labels=["sol1"])

    assert len(analyzer.df) == len(solution.bitstrings)
    assert "counts" in analyzer.df.columns
    assert "probs" in analyzer.df.columns


def test_analyzer_quantum(simple_qubo_instance: Instance) -> None:
    config = SolverConfig(use_quantum=True)
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()
    analyzer = Analyzer([solution], labels=["sol1"])

    assert len(analyzer.df) == len(solution.bitstrings)
    assert "probs" in analyzer.df.columns
    assert "counts" in analyzer.df.columns


@pytest.mark.parametrize("classical_method", [c.value for c in ClassicalSolverType])
def test_analyzer_quantum_and_classical(
    simple_qubo_instance: Instance, classical_method: str
) -> None:
    config = SolverConfig(
        use_quantum=False, classical=ClassicalConfig(classical_solver_type=classical_method)
    )
    solver = Solver(simple_qubo_instance, config)
    solution = solver.solve()

    quantumsolver = Solver(simple_qubo_instance, SolverConfig(use_quantum=True))
    quantumsolution = quantumsolver.solve()
    analyzer = Analyzer([solution, quantumsolution], labels=["sol1", "sol2"])

    assert len(analyzer.df) == len(solution.bitstrings) + len(quantumsolution.bitstrings)
