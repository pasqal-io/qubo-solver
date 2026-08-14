from __future__ import annotations

import torch
import numpy as np
import pytest
import pytest_check as check
from qubosolver import (
    Solver,
    Solution,
    SolverConfig,
    Dataset,
    Analyzer,
    Instance,
    solvers,
    Bitstring,
    bitstring,
    bitstrings,
    matrix,
    vector,
    vectori,
    utils,
    torch_rng,
)
from qubosolver.solvers.classical.bitflip import _bit_flip_local_search


@pytest.mark.parametrize("postprocessing", [True, False])
def test_basic_qubo_2d_integration(postprocessing: bool) -> None:

    # fmt: off
    Q = matrix.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = Instance(matrix=Q)
    solver = Solver(instance, SolverConfig(do_postprocessing=postprocessing))
    solution = Solution(
        bitstrings=bitstrings.tensor([[0, 0]]),
        costs=vector.tensor([0.0]),
        counts=vectori.tensor([1]),
        probabilities=vector.tensor([1.0]),
    )

    pp_solution = solver._post_process(solution)
    pp_solution._sort_by_cost()

    if postprocessing:
        torch.testing.assert_close(pp_solution.bitstrings[0, :], bitstring.tensor([1, 1]))
        torch.testing.assert_close(pp_solution.costs, vector.tensor([-18.0]))
    else:
        torch.testing.assert_close(pp_solution.bitstrings[0, :], bitstring.tensor([0, 0]))
        torch.testing.assert_close(pp_solution.costs, vector.tensor([0.0]))

    analyzer = Analyzer(pp_solution)
    df = analyzer.df
    print(f"\n{df}")


def test_basic_qubo_2d() -> None:

    # fmt: off
    Q = matrix.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = Instance(matrix=Q)
    solution = Solution(
        bitstrings=bitstrings.zeros(1, 2),
        costs=vector.zeros(1),
        counts=vectori.tensor([1]),
        probabilities=vector.tensor([1.0]),
    )

    pp_solution = solvers.iterative_bitflip_local_search(instance, solution)
    pp_solution._sort_by_cost()

    torch.testing.assert_close(pp_solution.bitstrings[0, :], bitstring.tensor([1, 1]))
    torch.testing.assert_close(pp_solution.costs, vector.tensor([-18.0]))

    analyzer = Analyzer(pp_solution)
    df = analyzer.df
    print(f"\n{df}")


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_random_qubos(density: float) -> None:

    size = 5

    for seed in [545, 87, 89993]:
        dataset = Dataset.from_random(1, size, densities=[density], rng=torch_rng(seed))
        torch.manual_seed(seed)
        for Q, _ in dataset:
            instance = Instance(matrix=Q)
            bitstring_ = (torch.rand(size) > 0.5).to(bitstring.dtype())
            cost = instance.cost(bitstring_)
            solution = Solution(
                bitstrings=bitstring_.unsqueeze(0),
                costs=vector.tensor([cost]),
                counts=vectori.tensor([1]),
                probabilities=vector.tensor([1.0]),
            )
            pp_solution = solvers.iterative_bitflip_local_search(instance, solution)
            pp_solution._sort_by_cost()

            analyzer = Analyzer(pp_solution)
            df = analyzer.df
            print(f"\n{df}")

            check.less_equal(pp_solution.costs[0], cost)


def test_no_solution() -> None:
    # fmt: off
    Q = matrix.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = Instance(matrix=Q)
    solution = Solution()
    check.equal(solution.bitstrings.numel(), 0)

    # Bitflip doesn't find new solutions if there were none to begin with
    pp_solution = solvers.iterative_bitflip_local_search(instance, solution)
    check.equal(pp_solution.bitstrings.numel(), 0)


@pytest.mark.parametrize("shuffle", [True, False])
def test_bit_flip_local_search_basic(shuffle: bool) -> None:

    # fmt: off
    Q = matrix.tensor([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on
    def cost_function(b: Bitstring) -> float:
        return utils._costs.quadratic_cost(b, Q)

    s = bitstring.zeros(2)
    initial_cost = cost_function(s)
    check.almost_equal(initial_cost, 0.0)

    best_bitstring, best_cost = _bit_flip_local_search(cost_function, s, rng=torch_rng(65))

    np.testing.assert_allclose(best_bitstring, np.array([1, 1]))
    check.almost_equal(best_cost, -18.0)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_bit_flip_local_search_randoms(shuffle: bool, density: float) -> None:

    size = 5

    for seed in [454, 85, 989751]:
        rng = torch_rng(seed)
        dataset = Dataset.from_random(1, size, densities=[density], rng=rng)
        s = bitstring.zeros(size)

        for Q, _ in dataset:

            def cost_function(b: Bitstring) -> float:
                return utils._costs.quadratic_cost(b, Q)

            initial_cost = cost_function(s)
            _, best_cost = _bit_flip_local_search(cost_function, s, rng=rng if shuffle else None)
            check.less_equal(best_cost, initial_cost)
