from __future__ import annotations

import pytest
import torch

from qubosolver import QUBOInstance, SolverConfig, QuboSolver, matrix, bitstrings, LocalEmulator


def test_classical_all_positive_trivial() -> None:
    """
    For a QUBO with all coefficients >= 0, the classical solver
    should return a batch of one all-zero bitstring
    with solution_status 'trivial-zero'.
    """
    coeffs = matrix.tensor([[1.0, 0.5], [0.5, 2.0]])
    instance = QUBOInstance(matrix=coeffs)
    config = SolverConfig(use_quantum=False)

    solver = QuboSolver(instance, config)
    sol = solver.solve()

    # All entries zero
    torch.testing.assert_close(sol.bitstrings, torch.zeros_like(sol.bitstrings))


def test_quantum_all_negative_trivial(local_backend: LocalEmulator) -> None:
    """
    For a QUBO with all coefficients <= 0, the quantum solver
    should return a batch of one all-one bitstring
    with solution_status 'trivial-one'.
    """

    config = SolverConfig(use_quantum=True, backend=local_backend)

    coeffs = matrix.tensor([[-1.0, -0.2], [-0.2, -3.0]])
    instance = QUBOInstance(matrix=coeffs)
    # check if value error is raised.
    with pytest.raises(
        ValueError, match="Quantum solver does not handle off-diagonal negative coefficients"
    ):
        solver = QuboSolver(instance, config)

    coeffs = matrix.tensor([[-1.0, 0.0], [0.0, -3.0]])
    instance = QUBOInstance(matrix=coeffs)

    solver = QuboSolver(instance, config)
    sol = solver.solve()

    # All entries one
    torch.testing.assert_close(sol.bitstrings, torch.ones_like(sol.bitstrings))


def test_diagonal_trivial(local_backend: LocalEmulator) -> None:
    coeffs = matrix.tensor([[-1.0, 0.0], [0.0, 3.0]])
    instance = QUBOInstance(matrix=coeffs)
    config = SolverConfig(use_quantum=True, backend=local_backend)

    solver = QuboSolver(instance, config)
    sol = solver.solve()
    torch.testing.assert_close(sol.bitstrings, bitstrings.tensor([[1, 0]]))
