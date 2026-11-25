# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check

from qubosolver import (
    Dataset,
    Instance,
    Solution,
    analysis,
    numpy_rng,
    solving,
)
from qubosolver.solving.classical.iterative_bitflip_local_search import (
    _best_improvement_search_batch,
)
from qubosolver.utils import _costs
from qubosolver.utils._random import manual_seed


def test_basic_qubo_2d() -> None:
    # fmt: off
    Q = np.asarray([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    instance = Instance(matrix=Q)
    solution = Solution(
        bitstrings=np.zeros((1, 2), dtype=np.int8),
        costs=np.zeros(1),
        counts=np.asarray([1]),
        probabilities=np.asarray([1.0]),
    )

    pp_solution = solving.iterative_bitflip_local_search.solve(instance, starts=solution)
    pp_solution._sort_by_cost()

    np.testing.assert_allclose(
        pp_solution.bitstrings[0, :], np.asarray([1, 1], dtype=np.int8), rtol=0, strict=True
    )
    np.testing.assert_allclose(pp_solution.costs, np.asarray([-18.0]), strict=True)

    df = analysis.to_dataframe([pp_solution])
    print(f"\n{df}")


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_random_qubos(density: float) -> None:

    size = 5

    for seed in [545, 87, 89993]:
        dataset = Dataset.from_random(1, size, densities=[density], rng=numpy_rng(seed))
        manual_seed(seed)
        for instance, _ in dataset:
            bitstring_ = (np.random.rand(size) > 0.5).astype(np.int8)
            cost = instance.cost(bitstring_)
            solution = Solution(
                bitstrings=bitstring_[np.newaxis],
                costs=np.asarray([cost], dtype=np.float64),
                counts=np.asarray([1]),
                probabilities=np.asarray([1.0]),
            )
            pp_solution = solving.iterative_bitflip_local_search.solve(instance, starts=solution)
            pp_solution._sort_by_cost()

            df = analysis.to_dataframe([pp_solution])
            print(f"\n{df}")

            check.less_equal(pp_solution.costs[0], cost)


def test_best_improvement_search_basic() -> None:
    # fmt: off
    Q = np.asarray([
        [-10.0, 1.0],
        [1.0, -10.0]
    ])
    # fmt: on

    s = np.zeros(2, dtype=np.int8)
    initial_cost = _costs.quadratic_cost(s, Q)
    check.almost_equal(initial_cost, 0.0)

    best_bitstrings = _best_improvement_search_batch(Q, s[np.newaxis])
    best_bitstring = best_bitstrings.squeeze(0)
    best_cost = _costs.quadratic_cost(best_bitstring, Q)

    np.testing.assert_allclose(best_bitstring, np.array([1, 1]))
    check.almost_equal(best_cost, -18.0)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_best_improvement_search_randoms(density: float) -> None:

    size = 5

    for seed in [454, 85, 989751]:
        rng = numpy_rng(seed)
        dataset = Dataset.from_random(1, size, densities=[density], rng=rng)
        s = np.zeros(size, dtype=np.int8)

        for instance, _ in dataset:
            initial_cost = _costs.quadratic_cost(s, instance.matrix)
            best_bitstrings = _best_improvement_search_batch(instance.matrix, s[np.newaxis])
            best_cost = _costs.quadratic_cost(best_bitstrings.squeeze(0), instance.matrix)
            check.less_equal(best_cost, initial_cost)
