from __future__ import annotations

import pytest
import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.config import ClassicalConfig, SolverConfig
from qubosolver.solver import QuboSolver


def test_qubo_solver_dwave_SA() -> None:
    # Create a simple 2x2 QUBO instance.
    # For example, consider a QUBO where the optimum is known.
    # Here we use an identity matrix.
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = QUBOInstance(coefficients=Q)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(classical_solver_type="dwave_sa")
    config = SolverConfig(use_quantum=False, classical=classical_config)

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = QuboSolver(instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of QUBOSolution.
    assert isinstance(solution, QUBOSolution)

    # The optimal value for binary variables is achieved when both are 0,
    # so expect a cost of 0.
    # Also, check that the bitstring has the expected shape, e.g., [1,2].
    assert solution.bitstrings.shape[0] == 1  # one solution returned
    assert solution.bitstrings.shape[1] == 2  # two variables

    # Additionally, check that the cost tensor is 0 (or very near to 0).
    expected_cost = 0.0
    actual_cost = solution.costs.item()  # convert cost tensor to a python float
    assert pytest.approx(actual_cost, rel=1e-3) == expected_cost


def test_qubo_solver_dwave_tabu() -> None:
    # Create a simple 2x2 QUBO instance.
    # For example, consider a QUBO where the optimum is known.
    # Here we use an identity matrix.
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = QUBOInstance(coefficients=Q)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(classical_solver_type="dwave_tabu")
    config = SolverConfig(use_quantum=False, classical=classical_config)

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = QuboSolver(instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of QUBOSolution.
    assert isinstance(solution, QUBOSolution)

    # The optimal value for binary variables is achieved when both are 0,
    # so expect a cost of 0.
    # Also, check that the bitstring has the expected shape, e.g., [1,2].
    assert solution.bitstrings.shape[0] == 1  # one solution returned
    assert solution.bitstrings.shape[1] == 2  # two variables

    # Additionally, check that the cost tensor is 0 (or very near to 0).
    expected_cost = 0.0
    actual_cost = solution.costs.item()  # convert cost tensor to a python float
    assert pytest.approx(actual_cost, rel=1e-3) == expected_cost


if __name__ == "__main__":
    pytest.main()
