# tests/test_qubo_instance.py
from __future__ import annotations

import os
from pathlib import Path
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

    qi.set_coefficients()
    assert qi.coefficients.shape == (5, 5)
    qi.set_coefficients({(i, i): 2.0 for i in range(5)})
    assert qi.coefficients.shape == (5, 5)
    for i in range(5):
        assert qi.coefficients[i, i] == 2.0

    qi._expand_size(10)
    assert qi.coefficients.shape == (10, 10)


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


def test_save_load(simple_qubo_instance: QUBOInstance) -> None:

    file_path = Path(__file__).parent / "qubo_instance_test.pt"
    QUBOInstance.save(file_path, simple_qubo_instance)
    assert os.path.exists(file_path)
    loaded_instance = QUBOInstance.load(file_path)
    assert torch.allclose(loaded_instance.coefficients, simple_qubo_instance.coefficients)
    if os.path.exists(file_path):
        os.remove(file_path)
