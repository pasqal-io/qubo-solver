from __future__ import annotations

import pytest
import pytest_check as check

from qubosolver import QUBOSolution, Solution, QuboSolution


def test_qubo_solution_wrong_case_deprecation() -> None:
    with pytest.warns(DeprecationWarning, match="Use `qubosolver.Solution` instead"):
        solution = QUBOSolution()
    check.is_instance(solution, Solution)
    check.is_false(solution)

def test_qubo_solution_alias() -> None:
    solution = QuboSolution()
    check.is_instance(solution, Solution)
    check.is_false(solution)
