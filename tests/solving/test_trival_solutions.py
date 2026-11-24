# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import pytest_check as check
import torch

from qubosolver import Instance, bitstrings, matrix, solving


def test_all_positive_trivial() -> None:
    """For a QUBO with all coefficients >= 0, the trivial solution search returns all zeros."""
    coeffs = matrix.tensor([[1.0, 0.5], [0.5, 2.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)

    # All entries zero
    torch.testing.assert_close(sol.bitstrings, torch.zeros_like(sol.bitstrings))
    check.is_true(sol.check_consistency())


def test_all_negative_trivial() -> None:
    """For a QUBO with all coefficients <= 0, the trivial solution search returns all ones."""
    coeffs = matrix.tensor([[-1.0, 0.0], [0.0, -3.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)

    # All entries one
    torch.testing.assert_close(sol.bitstrings, torch.ones_like(sol.bitstrings))
    check.is_true(sol.check_consistency())


def test_diagonal_trivial() -> None:
    coeffs = matrix.tensor([[-1.0, 0.0], [0.0, 3.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)
    torch.testing.assert_close(sol.bitstrings, bitstrings.tensor([[1, 0]]))
    check.is_true(sol.check_consistency())
