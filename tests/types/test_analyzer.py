# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qoolqit

from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    drive_shaping,
    embedding,
    solving,
)
from qubosolver.utils import analysis

if TYPE_CHECKING:
    from types import ModuleType

_CLASSICAL_METHODS = [
    solving.tabu_search,
    solving.simulated_annealing,
    pytest.param(solving.cplex, marks=pytest.mark.extras),
    solving.random_sampling,
]
_CLASSICAL_IDS = ["tabu_search", "simulated_annealing", "cplex", "random_sampling"]


def _quantum_solve(instance: Instance) -> Solution:
    device = qoolqit.AnalogDeviceWithDMM()
    register = embedding.blade.embed_for_device(instance, device)
    drive = drive_shaping.local_energy_scale.build_drive(instance, register, device=device)
    program = solving.analog_quantum_sampling.compile(register, drive, device)
    return Solution.from_results(LocalEmulator().run(program).results(), instance)


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


def test_to_dataframe_counts_are_stored_as_ints(basic_solution: Solution) -> None:
    df = analysis.to_dataframe([basic_solution])
    assert df["counts"].dtype == "int64"


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
    filtered = analysis._filter_by_percentage(
        df, top_percent=0.5, column="probs", order="descending"
    )
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


@pytest.mark.parametrize("classical_method", _CLASSICAL_METHODS, ids=_CLASSICAL_IDS)
def test_analyzer_classical(simple_qubo_instance: Instance, classical_method: ModuleType) -> None:
    solution = classical_method.solve(simple_qubo_instance)
    df = analysis.to_dataframe([solution], labels=["sol1"])

    assert len(df) == len(solution.bitstrings)
    assert "counts" in df.columns
    assert "probs" in df.columns


@pytest.mark.duration(4)
def test_analyzer_quantum(simple_qubo_instance: Instance) -> None:
    solution = _quantum_solve(simple_qubo_instance)
    df = analysis.to_dataframe([solution], labels=["sol1"])

    assert len(df) == len(solution.bitstrings)
    assert "probs" in df.columns
    assert "counts" in df.columns


@pytest.mark.duration(4)
@pytest.mark.parametrize("classical_method", _CLASSICAL_METHODS, ids=_CLASSICAL_IDS)
def test_analyzer_quantum_and_classical(
    simple_qubo_instance: Instance, classical_method: ModuleType
) -> None:
    solution = classical_method.solve(simple_qubo_instance)
    quantumsolution = _quantum_solve(simple_qubo_instance)
    df = analysis.to_dataframe([solution, quantumsolution], labels=["sol1", "sol2"])

    assert len(df) == len(solution.bitstrings) + len(quantumsolution.bitstrings)
