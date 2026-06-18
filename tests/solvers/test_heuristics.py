from __future__ import annotations

import pytest
import torch
import itertools

import time

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


@pytest.mark.parametrize(
    "classical_methods",
    [ClassicalSolverType.SIMULATED_ANNEALING, ClassicalSolverType.SIMULATED_ANNEALING_TABU_SEARCH],
)
@pytest.mark.parametrize("max_bitstrings", [1])
def test_sa_cost(
    simple_qubo_instance: QUBOInstance, classical_methods: ClassicalSolverType, max_bitstrings: int
) -> None:
    classical_config = ClassicalConfig(
        classical_solver_type=classical_methods, max_bitstrings=max_bitstrings, sa_seed=42
    )

    config = SolverConfig(
        use_quantum=False, classical=classical_config, activate_trivial_solutions=False
    )

    assert isinstance(
        get_classical_solver(simple_qubo_instance, config.classical),
        class_solvers[classical_methods],
    )

    classical_solver = QuboSolver(simple_qubo_instance, config)
    solution = classical_solver.solve()

    assert isinstance(solution, QUBOSolution)

    Q = simple_qubo_instance.coefficients
    n = Q.shape[0]

    bitstrings = []
    costs = []

    for bits in itertools.product([0, 1], repeat=n):
        z = torch.tensor(bits, dtype=torch.float32)
        bitstrings.append("".join(map(str, bits)))
        zQz = z @ Q @ z
        costs.append(zQz.item())

    sorted_results = sorted(zip(bitstrings, costs), key=lambda x: x[1])
    bests = [(b, c) for b, c in sorted_results[:max_bitstrings]]

    for bitstring_, cost_ in bests:
        bitstring = torch.tensor(list(map(int, bitstring_)), dtype=torch.int32)
        cost = torch.tensor(float(cost_), dtype=torch.float32)

        bitstring_sa = solution.bitstrings[0]
        cost_sa = solution.costs[0]

        assert torch.equal(bitstring, bitstring_sa)
        cost_sa = cost_sa.to(dtype=cost.dtype)
        assert torch.isclose(cost, cost_sa, rtol=1e-4)


def test_sa_time_limit(simple_qubo_instance: QUBOInstance) -> None:
    # Use a very large iteration limit so that the solver is stopped
    # by the time limit rather than by max_iter.
    classical_config = ClassicalConfig(
        classical_solver_type=ClassicalSolverType.SIMULATED_ANNEALING,
        max_bitstrings=1,
        max_iter=100_000_000,
        sa_time_limit=0.01,
    )

    config = SolverConfig(
        use_quantum=False,
        classical=classical_config,
        activate_trivial_solutions=False,
    )

    classical_solver = QuboSolver(simple_qubo_instance, config)

    # Measure the full execution time of the solver.
    start_time = time.perf_counter()
    solution = classical_solver.solve()
    elapsed_time = time.perf_counter() - start_time

    # Check that a valid solution is returned and that the solver
    # stops well before reaching the large iteration limit.
    assert isinstance(solution, QUBOSolution)
    assert elapsed_time < 1.0


if __name__ == "__main__":
    pytest.main()
