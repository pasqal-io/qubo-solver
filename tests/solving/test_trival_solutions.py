# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import numpy as np
import pytest_check as check

from qubosolver import Instance, solving


def test_all_positive_trivial() -> None:
    """For a QUBO with all coefficients >= 0, the trivial solution search returns all zeros."""
    coeffs = np.asarray([[1.0, 0.5], [0.5, 2.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)

    # All entries zero
    np.testing.assert_allclose(sol.bitstrings, np.zeros_like(sol.bitstrings), rtol=0, strict=True)
    check.is_true(sol.check_consistency())


def test_all_negative_trivial() -> None:
    """For a QUBO with all coefficients <= 0, the trivial solution search returns all ones."""
    coeffs = np.asarray([[-1.0, 0.0], [0.0, -3.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)

    # All entries one
    np.testing.assert_allclose(sol.bitstrings, np.ones_like(sol.bitstrings), rtol=0, strict=True)
    check.is_true(sol.check_consistency())


def test_diagonal_trivial() -> None:
    coeffs = np.asarray([[-1.0, 0.0], [0.0, 3.0]])
    instance = Instance(matrix=coeffs)

    sol = solving.trivial_solution_search.solve(instance)
    np.testing.assert_allclose(
        sol.bitstrings, np.asarray([[1, 0]], dtype=np.int8), rtol=0, strict=True
    )
    check.is_true(sol.check_consistency())
