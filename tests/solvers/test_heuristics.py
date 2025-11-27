from __future__ import annotations

import pytest
import torch

from qubosolver import QUBOInstance, QUBOSolution, ClassicalSolverType
from qubosolver.config import ClassicalConfig, SolverConfig
from qubosolver.solver import QuboSolver
from qubosolver.classical_solver import get_classical_solver
from qubosolver.classical_solver.classical_solver import (
    SimulatedAnnealingSolver,
    TabuSearchSolver,
    HybridSATabuSolver,
    RandomSolver,
)

class_solvers = {
    ClassicalSolverType.SIMULATED_ANNEALING: SimulatedAnnealingSolver,
    ClassicalSolverType.TABU_SEARCH: TabuSearchSolver,
    ClassicalSolverType.SIMULATED_ANNEALING_TABU_SEARCH: HybridSATabuSolver,
}


@pytest.mark.parametrize("classical_method", list(class_solvers.keys()))
@pytest.mark.parametrize("max_bitstrings", [1, 3])
def test_qubo_solver_sa_or_tabu(
    simple_qubo_instance: QUBOInstance, classical_method: ClassicalSolverType, max_bitstrings: int
) -> None:
    # Create a SolverConfig object with classical solver options.
    classical_config = ClassicalConfig(
        classical_solver_type=classical_method, max_bitstrings=max_bitstrings
    )
    config = SolverConfig(
        use_quantum=False, classical=classical_config, activate_trivial_solutions=False
    )

    # insure get_classical_solver works properly
    assert isinstance(
        get_classical_solver(simple_qubo_instance, config.classical),
        class_solvers[classical_method],
    )

    # Instantiate the classical solver via the pipeline's classical solver dispatcher.
    classical_solver = QuboSolver(simple_qubo_instance, config)

    # Solve the QUBO problem.
    solution = classical_solver.solve()

    # Assert that the solution is an instance of QUBOSolution.
    assert isinstance(solution, QUBOSolution)

    # assert shape matches config
    assert solution.bitstrings.shape[0] > 0
    assert solution.bitstrings.shape[0] <= max_bitstrings  # max_bitstrings solution returned
    assert solution.bitstrings.shape[1] == simple_qubo_instance.size
    assert solution.counts is not None
    assert solution.probabilities is not None
    assert solution.counts.sum().item() == max_bitstrings
    assert torch.allclose(solution.probabilities.sum(), torch.ones(1))


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
    assert solution.bitstrings.shape[1] == 2  # two variables
    assert len(solution.bitstrings) <= classical_config.max_bitstrings
    assert len(solution.costs) == len(solution.bitstrings)
    assert solution.counts is not None
    assert solution.probabilities is not None
    assert solution.counts.sum().item() == classical_config.max_bitstrings
    assert torch.allclose(solution.probabilities.sum(), torch.ones(1))


if __name__ == "__main__":
    pytest.main()
