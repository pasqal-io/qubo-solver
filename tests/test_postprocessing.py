from __future__ import annotations

import torch
import pytest
from qubosolver.solver import QuboSolver, QUBOSolution
from qubosolver.config import SolverConfig
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.pipeline.fixtures import Fixtures


@pytest.mark.parametrize("postprocessing", [True, False])
def test_basic_qubo_2d_integration(postprocessing: bool) -> None:

    # fmt: off
    Q = torch.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = QUBOInstance(coefficients=Q)
    solver = QuboSolver(instance, SolverConfig(do_postprocessing=postprocessing))
    solution = QUBOSolution(bitstrings=torch.tensor([[0, 0]]), costs=torch.tensor([0.0]))

    pp_solution = solver.post_process(solution)
    pp_solution.sort_by_cost()
    pp_solution.bitstrings = pp_solution.bitstrings.int()

    if postprocessing:
        torch.testing.assert_close(
            pp_solution.bitstrings[0, :], torch.tensor([1, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(pp_solution.costs, torch.tensor([-18.0]))
    else:
        torch.testing.assert_close(
            pp_solution.bitstrings[0, :], torch.tensor([0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(pp_solution.costs, torch.tensor([0.0]))

    analyzer = QUBOAnalyzer(pp_solution)
    df = analyzer.df
    print(f"\n{df}")


@pytest.mark.parametrize("postprocessing", [True, False])
def test_basic_qubo_2d_fixture(postprocessing: bool) -> None:

    # fmt: off
    Q = torch.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = QUBOInstance(coefficients=Q)
    fixture = Fixtures(instance, SolverConfig(do_postprocessing=postprocessing))
    solution = QUBOSolution(bitstrings=torch.tensor([[0, 0]]), costs=torch.tensor([0.0]))

    pp_solution = fixture.postprocess(solution)
    pp_solution.sort_by_cost()
    pp_solution.bitstrings = pp_solution.bitstrings.int()

    if postprocessing:
        torch.testing.assert_close(
            pp_solution.bitstrings[0, :], torch.tensor([1, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(pp_solution.costs, torch.tensor([-18.0]))
    else:
        torch.testing.assert_close(
            pp_solution.bitstrings[0, :], torch.tensor([0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(pp_solution.costs, torch.tensor([0.0]))

    analyzer = QUBOAnalyzer(pp_solution)
    df = analyzer.df
    print(f"\n{df}")
