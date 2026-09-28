from __future__ import annotations

import pytest
import torch

from qubosolver import QUBOInstance
from qubosolver.config import BackendConfig, SolverConfig
from qubosolver.solver import QuboSolverClassical, QuboSolverQuantum


def test_classical_all_positive_trivial() -> None:
    """
    For a QUBO with all coefficients >= 0, the classical solver
    should return a batch of one all-zero bitstring
    with solution_status 'trivial-zero'.
    """
    coeffs = [[1.0, 0.5], [0.5, 2.0]]
    instance = QUBOInstance(coefficients=coeffs)
    config = SolverConfig(use_quantum=False)

    solver = QuboSolverClassical(instance, config)
    sol = solver.solve()

    # All entries zero
    assert torch.all(sol.bitstrings == 0), f"Expected all zeros, got {sol.bitstrings}"
    # Status should indicate trivial-zero
    assert hasattr(sol, "solution_status"), "QUBOSolution missing 'solution_status' attribute"
    assert (
        sol.solution_status == "trivial-zero"
    ), f"Expected status 'trivial-zero', got {sol.solution_status}"


def test_quantum_all_negative_trivial(local_backend: BackendConfig) -> None:
    """
    For a QUBO with all coefficients <= 0, the quantum solver
    should return a batch of one all-one bitstring
    with solution_status 'trivial-one'.
    """
    coeffs = [[-1.0, -0.2], [-0.2, -3.0]]
    # check if value error is raised.
    with pytest.raises(ValueError, match="Negative off-diagonal coefficient"):
        QUBOInstance(coefficients=coeffs)

    coeffs = [[-1.0, 0.0], [0.0, -3.0]]
    instance = QUBOInstance(coefficients=coeffs)
    config = SolverConfig(use_quantum=True, backend_config=local_backend)

    solver = QuboSolverQuantum(instance, config)
    sol = solver.solve()

    # All entries one
    assert torch.all(sol.bitstrings == 1), f"Expected all ones, got {sol.bitstrings}"
    # Status should indicate trivial-one
    assert hasattr(sol, "solution_status"), "QUBOSolution missing 'solution_status' attribute"
    assert (
        sol.solution_status == "trivial-one"
    ), f"Expected status 'trivial-one', got {sol.solution_status}"
