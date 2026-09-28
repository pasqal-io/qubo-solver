# tests/test_qubo_instance.py
from __future__ import annotations

import numpy as np
import pytest
import torch

from qubosolver.qubo_instance import QUBOInstance


def test_valid_qubo_passes_without_error() -> None:
    # A 5×5 QUBO with all coefficients >= 0 (identity matrix)
    coeffs = torch.eye(5)
    qi = QUBOInstance()
    # Should not raise any exception
    qi.coefficients = coeffs
    assert qi.size == 5
    # Verify that the tensor is stored correctly
    assert qi.coefficients.shape == (5, 5)


def test_negative_off_diagonal_triggers_system_exit() -> None:
    # A 3×3 QUBO with a negative off-diagonal coefficient at (0,1)
    coeffs = np.zeros((3, 3))
    coeffs[0, 1] = -1.0

    qi = QUBOInstance()
    # Expect SystemExit to be raised with the correct error message
    msg = "Error: Negative off-diagonal coefficient detected."
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        qi.coefficients = coeffs


def test_size_exceeds_limit_triggers_system_exit() -> None:
    from qubosolver.solver import QuboSolverClassical, QuboSolverQuantum

    # An 81×81 QUBO exceeds the maximum supported size of 80×80
    coeffs = np.zeros((81, 81))

    qi = QUBOInstance()
    qi.coefficients = coeffs
    # Expect SystemExit to be raised when setting oversized coefficients
    solver = QuboSolverClassical(qi)
    assert solver.instance.size == 81
    match_msg = "QUBO size 81×81 exceeds the maximum supported size of 80×80"
    with pytest.raises(ValueError, match=match_msg):
        QuboSolverQuantum(qi)
