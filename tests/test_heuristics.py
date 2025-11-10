from __future__ import annotations

import pytest
import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.config import ClassicalConfig, SolverConfig
from qubosolver.solver import QuboSolver
from qubosolver.classical_solver import get_classical_solver
from qubosolver.classical_solver.classical_solver import (
    SimulatedAnnealingSolver,
    TabuSearchSolver,
    RandomSolver,
)


@pytest.mark.parametrize("max_bitstrings", [1, 3])
def test_qubo_solver_sa(simple_qubo_instance: QUBOInstance, max_bitstrings: int) -> None:
    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(
        classical_solver_type="simulated_annealing", max_bitstrings=max_bitstrings
    )
    config = SolverConfig(
        use_quantum=False, classical=classical_config, activate_trivial_solutions=False
    )

    # insure get_classical_solver works properly
    assert isinstance(
        get_classical_solver(simple_qubo_instance, config.classical), SimulatedAnnealingSolver
    )

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = QuboSolver(simple_qubo_instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of QUBOSolution.
    assert isinstance(solution, QUBOSolution)

    # The optimal value for binary variables is achieved when both are 0,
    # so expect a cost of 0.
    # Also, check that the bitstring has the expected shape, e.g., [1,2].
    assert (
        solution.bitstrings.shape[0] <= max_bitstrings
    )  # at most max_bitstrings solution returned
    assert solution.bitstrings.shape[1] == 3  # three variables
    assert solution.counts is not None
    assert solution.probabilities is not None
    assert solution.counts.sum().item() == max_bitstrings
    assert torch.allclose(solution.probabilities.sum(), torch.ones(1))


def test_qubo_solver_tabu() -> None:
    # Create a simple 2x2 QUBO instance.
    # For example, consider a QUBO where the optimum is known.
    # Here we use an identity matrix.
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = QUBOInstance(coefficients=Q)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(classical_solver_type="tabu_search")
    config = SolverConfig(
        use_quantum=False, classical=classical_config, activate_trivial_solutions=False
    )

    assert isinstance(get_classical_solver(instance, config.classical), TabuSearchSolver)

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


def test_random() -> None:
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    instance = QUBOInstance(coefficients=Q)

    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(classical_solver_type="random", max_bitstrings=3)
    config = SolverConfig(
        use_quantum=False, classical=classical_config, activate_trivial_solutions=False
    )

    # insure get_classical_solver works properly
    assert isinstance(get_classical_solver(instance, config.classical), RandomSolver)

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = QuboSolver(instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of QUBOSolution.
    assert isinstance(solution, QUBOSolution)
    assert solution.bitstrings.shape[0] == classical_config.max_bitstrings
    assert len(solution.costs) == classical_config.max_bitstrings
    assert solution.bitstrings.shape[1] == 2  # two variables


if __name__ == "__main__":
    pytest.main()
