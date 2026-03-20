from __future__ import annotations

import torch
import numpy as np
import pytest
import pytest_check as check
from qubosolver.solver import QuboSolver, QUBOSolution
from qubosolver.config import SolverConfig
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.pipeline.fixtures import Fixtures, bit_flip_local_search
from qubosolver.data import QUBODataset


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


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("postprocessing", [True, False])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_random_qubos(postprocessing: bool, density: float) -> None:

    size = 5

    for seed in [545, 87, 89993]:
        dataset = QUBODataset.from_random(1, size, densities=[density], seed=seed)
        torch.manual_seed(seed)
        for Q, _ in dataset:
            instance = QUBOInstance(coefficients=Q)
            bitstring = (torch.rand(size) > 0.5).int()
            cost = instance.evaluate_solution(bitstring)
            fixture = Fixtures(instance, SolverConfig(do_postprocessing=postprocessing))
            solution = QUBOSolution(bitstrings=bitstring.unsqueeze(0), costs=torch.tensor([cost]))
            pp_solution = fixture.postprocess(solution)
            pp_solution.sort_by_cost()
            pp_solution.bitstrings = pp_solution.bitstrings.int()

            analyzer = QUBOAnalyzer(pp_solution)
            df = analyzer.df
            print(f"\n{df}")

            if postprocessing:
                check.less_equal(pp_solution.costs[0], cost)
            else:
                check.almost_equal(pp_solution.costs[0], cost)
                torch.testing.assert_close(pp_solution.bitstrings[0, :], bitstring)


def test_no_solution() -> None:
    # fmt: off
    Q = torch.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = QUBOInstance(coefficients=Q)
    fixture = Fixtures(instance, SolverConfig(do_postprocessing=True))
    solution = QUBOSolution(bitstrings=torch.tensor([]), costs=torch.tensor([]))
    check.equal(solution.bitstrings.numel(), 0)

    # Post-processing doesn't find new solutions if there were none to begin with
    pp_solution = fixture.postprocess(solution)
    check.equal(pp_solution.bitstrings.numel(), 0)


@pytest.mark.parametrize("shuffle", [True, False])
def test_bit_flip_local_search_basic(shuffle: bool) -> None:

    # fmt: off
    Q = torch.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on
    def cost_function(bitstring: np.ndarray) -> float:
        return float(bitstring.T @ Q.numpy() @ bitstring)

    s = np.zeros(2)
    initial_cost = cost_function(s)
    check.almost_equal(initial_cost, 0.0)

    best_bitstring, best_cost = bit_flip_local_search(cost_function, s, shuffle=shuffle)

    np.testing.assert_allclose(best_bitstring, np.array([1, 1]))
    check.almost_equal(best_cost, -18.0)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_bit_flip_local_search_randoms(shuffle: bool, density: float) -> None:

    size = 5

    for seed in [454, 85, 989751]:
        dataset = QUBODataset.from_random(1, size, densities=[density], seed=seed)
        s = np.zeros(size)

        for Q, _ in dataset:

            def cost_function(bitstring: np.ndarray) -> float:
                return float(bitstring.T @ Q.numpy() @ bitstring)

            initial_cost = cost_function(s)
            _, best_cost = bit_flip_local_search(cost_function, s, shuffle=shuffle)
            check.less_equal(best_cost, initial_cost)
