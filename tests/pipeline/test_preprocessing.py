# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from copy import deepcopy

import pytest_check as check
import torch

from qubosolver import (
    Instance,
    Solution,
    bitstring,
    bitstrings,
    matrix,
    transforms,
    vector,
    vectori,
)


def test_apply_full_and_post_process_fixation() -> None:

    instance = Instance(
        matrix.tensor(
            [
                [-98, 2, 13, 1],
                [2, -12, 20, 15],
                [13, 20, -34, 7],
                [1, 15, 7, -57],
            ]
        )
    )

    reduced_instance = transforms.variable_fixing.apply_recursively(instance)

    assert isinstance(reduced_instance, transforms.variable_fixing.Instance)
    check.equal(reduced_instance.fixed_history, [{0: 1, 3: 1}, {1: 0, 2: 0}])
    check.equal(len(reduced_instance.fixed_variables), 4)

    reduced_solution = Solution(
        bitstrings.zeros(1, 0),
        vector.tensor([0.0]),
        vectori.tensor([1]),
        vector.tensor([1.0]),
    )

    solution = transforms.variable_fixing.lift(reduced_solution, reduced_instance)

    check.equal(len(solution), 1)
    check.equal(solution[0].cost, -153)


def test_hansen_fixing() -> None:
    matrix_reducible = matrix.tensor([[-10, 1], [1, -10]])

    qubo_reducible = Instance(matrix_reducible)

    fixed_var = transforms.variable_fixing.hansen_fixing(qubo_reducible)

    assert isinstance(fixed_var, dict)

    assert fixed_var == {0: 1, 1: 1}

    matrix_not_reducible = matrix.tensor([[-1, 10], [10, -1]])

    qubo_reducible = Instance(matrix_not_reducible)

    empty_fixed_var = transforms.variable_fixing.hansen_fixing(qubo_reducible)

    assert empty_fixed_var == {}


def test_reduce_qubo() -> None:

    Q = matrix.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
    )
    qubo = Instance(Q)
    reduced_qubo = deepcopy(qubo)

    reduced_qubo = transforms.variable_fixing._reduce_qubo(qubo, {0: 1, 3: 1})

    expected_coefficients = matrix.tensor([[22, 20], [20, 6]])
    torch.testing.assert_close(reduced_qubo.matrix, expected_coefficients)

    non_reduced_qubo = transforms.variable_fixing._reduce_qubo(qubo, {})
    torch.testing.assert_close(non_reduced_qubo.matrix, qubo.matrix)


def test_apply_rule() -> None:

    Q = matrix.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
    )
    qubo = Instance(Q)
    reduced_qubo = transforms.variable_fixing.apply(
        qubo, [transforms.variable_fixing.hansen_fixing]
    )

    expected_coefficients = matrix.tensor([[22, 20], [20, 6]])
    torch.testing.assert_close(reduced_qubo.matrix, expected_coefficients)


def test_reduce_qubo_2() -> None:

    Q = matrix.tensor(
        [
            [1.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ],
    )

    qubo = Instance(Q)

    reduced_qubo = transforms.variable_fixing.apply_recursively(qubo)

    check.equal(reduced_qubo.fixed_history, [{0: 0}, {3: 1, 4: 1}])

    # Hardcode solution
    reduced_solution = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([0.0]),
    )

    solution = transforms.variable_fixing.lift(reduced_solution, reduced_qubo)
    check.equal(solution.bitstrings.shape, (1, 5))

    bitstring_ = bitstring.to_string(solution.bitstrings[0])
    check.equal(bitstring_, "00111")
    check.almost_equal(solution.costs[0], -27.288260)
