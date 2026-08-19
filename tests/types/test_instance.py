# tests/test_qubo_instance.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import pytest_check as check
import torch

from qubosolver import Instance, Solver, solvers, matrix


def test_valid_qubo_passes_without_error() -> None:
    # A 5×5 QUBO with all coefficients >= 0 (identity matrix)
    coeffs = matrix.from_torch(torch.eye(5))
    qi = Instance(coeffs)
    assert qi.size == 5
    # Verify that the tensor is stored correctly
    assert qi.matrix.shape == (5, 5)


def test_len_matches_size() -> None:
    qi = Instance(matrix.from_torch(torch.eye(5)))
    check.equal(len(qi), 5)
    check.is_true(bool(qi))


def test_len_zero_is_falsy() -> None:
    qi = Instance(matrix.zeros(0))
    check.equal(len(qi), 0)
    check.is_false(bool(qi))


def test_size_exceeds_limit_triggers_system_exit() -> None:

    # An 81×81 QUBO exceeds the maximum supported size of 80×80
    coeffs = np.zeros((81, 81))

    qi = Instance(matrix.tensor(coeffs))
    # Expect SystemExit to be raised when setting oversized coefficients
    solver = Solver(qi, solvers.Config(solving=solvers.ClassicalConfig()))
    assert solver.instance.size == 81
    match_msg = "QUBO size 81×81 exceeds the maximum supported size of 80×80"
    with pytest.raises(ValueError, match=match_msg):
        Solver(qi, solvers.Config(solving=solvers.QuantumConfig()))


@pytest.mark.parametrize("size", [0, 1])
def test_max_off_diag_no_off_diag_entries(size: int) -> None:
    qi = Instance(matrix.zeros(size))
    with pytest.raises(RuntimeError, match="undefined"):
        qi._max_off_diag


def test_save_load(simple_qubo_instance: Instance) -> None:

    file_path = Path(__file__).parent / "qubo_instance_test.pt"
    Instance.save(file_path, simple_qubo_instance)
    assert os.path.exists(file_path)
    loaded_instance = Instance.load(file_path)
    assert torch.allclose(loaded_instance.matrix, simple_qubo_instance.matrix)
    if os.path.exists(file_path):
        os.remove(file_path)
