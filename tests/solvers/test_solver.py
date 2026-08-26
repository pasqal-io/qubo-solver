from __future__ import annotations

import logging

import numpy as np
import pytest

import pytest_check as check
import torch
from unittest.mock import Mock

from scipy.spatial.distance import pdist, squareform

from qoolqit import Device
from qoolqit.execution import JobStatus
from qoolqit.graphs import DataGraph

from qubosolver import (
    Instance,
    Solver,
    Solution,
    embedding,
    solvers,
    drive_shaping,
    vectori,
    bitstrings,
    matrix,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.utils import analysis
from qubosolver.solvers.solver import _QuboSolverQuantum
from qubosolver.drive_shaping import Algorithm
from mock.connection import MockConnection

from pulser.backend.remote import (
    Results,
    RemoteConnection,
)
from emu_sv import SVBackend

from typing import Optional


@pytest.fixture
def implicit_default_qubo_solver_config(
    simple_qubo_instance: Instance,
) -> Solver:
    default_solver = Solver(simple_qubo_instance)
    return default_solver


def test_implicit_solver_config(
    implicit_default_qubo_solver_config: Solver,
) -> None:
    assert isinstance(implicit_default_qubo_solver_config._solver, _QuboSolverQuantum)


def test_different_shots(simple_qubo_instance: Instance) -> None:
    from pulser_simulation import QutipBackendV2

    default_solver = Solver(
        simple_qubo_instance,
        solvers.Config(
            solving=solvers.quantum.Config(
                backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500)
            )
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500

    lessshots_solver = Solver(
        simple_qubo_instance,
        solvers.Config(
            solving=solvers.quantum.Config(
                backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=100)
            )
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100


@pytest.mark.priority(40)
@pytest.mark.flaky(max_runs=5)
def test_run_local_backends(simple_qubo_instance: Instance, local_backend: LocalEmulator) -> None:
    solver = Solver(
        simple_qubo_instance,
        solvers.Config(
            solving=solvers.quantum.Config(
                backend=local_backend,
                embedding=embedding.Config(algorithm=embedding.Algorithm.BLADE),
            )
        ),
    )
    solutions = solver.solve()
    # theoretically -4.4000 can be found
    assert solutions.costs.min().item() <= -3.0


@pytest.mark.priority(150)
def test_solver_different_devices(
    request: pytest.FixtureRequest,
    qubo_for_testing_many_devices: Instance,
    local_device: Device,
    embedding_algorithm: embedding.Algorithm,
) -> None:

    quantum_config = solvers.quantum.Config(
        drive_shaping=drive_shaping.Config(algorithm="proportional_diagonal"),
        embedding=embedding.Config(
            algorithm=embedding_algorithm,
            greedy_layout_traps=qubo_for_testing_many_devices.size,
        ),
        device=local_device,
        backend=LocalEmulator(backend_type=SVBackend),

    )
    config = solvers.Config(
        solving=quantum_config,
        do_postprocessing=False,
        do_preprocessing=False,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    solution = solver.solve()
    assert solution


def test_parse_results() -> None:
    # Mock results object
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"001": 10, "110": 5, "010": 3}

    solution = Solution.from_results(mock_result, Instance())

    expected_bitstrings = bitstrings.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    expected_counts = vectori.tensor([10, 5, 3])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_empty_final_bitstrings() -> None:
    # Mock results with empty final_bitstrings
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {}

    solution = Solution.from_results(mock_result, Instance())

    check.equal(solution.bitstrings.shape, (0, 0))
    check.equal(solution.bitstrings.dtype, torch.int8)
    check.equal(solution.counts.shape, (0,))
    check.equal(solution.counts.dtype, torch.int64)


def test_parse_results_binary_string_conversion() -> None:
    # Mock results with binary string keys
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"0101": 8, "1010": 12, "1111": 4}

    solution = Solution.from_results(mock_result, Instance())

    expected_bitstrings = bitstrings.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    expected_counts = vectori.tensor([8, 12, 4])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_single_bitstring() -> None:
    # Mock results with single bitstring
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": 25}

    solution = Solution.from_results(mock_result, Instance())

    expected_bitstrings = bitstrings.tensor([[1, 0, 1]])
    expected_counts = vectori.tensor([25])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_string_counts_to_integer_tensor() -> None:
    # Mock results with string count values
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": "15", "010": "8", "111": "12"}

    solution = Solution.from_results(mock_result, Instance())

    expected_bitstrings = bitstrings.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    expected_counts = vectori.tensor([15, 8, 12])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def trivial_triangular_qubo(connection: Optional[RemoteConnection] = None) -> Solver:
    Q = 10.0 * matrix.tensor(
        [
            [-10.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    qubo = Instance(Q)

    num_shots = 100
    backend = (
        LocalEmulator(num_shots=num_shots)
        if connection is None
        else RemoteEmulator(connection=connection, num_shots=num_shots)
    )
    config = solvers.Config(
        solving=solvers.quantum.Config(
            embedding=embedding.Config(algorithm="blade"),
            backend=backend,
        ),
        do_preprocessing=False,
    )

    solver = Solver(qubo, config)

    return solver


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("wait", [True, False], ids=["wait", "dont_wait"])
def test_submit_integration(make_mock_connection: type[MockConnection], wait: bool) -> None:
    seed = 16844214
    np.random.seed(seed)

    solver = trivial_triangular_qubo()

    embedding = solver._embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver._drive(embedding)

    job = solver._submit(drive, embedding)
    results = job.results()

    solution = Solution.from_results(results, solver.instance)

    # Take the top 3 solutions with the highest probabilities
    sorted_indices = torch.argsort(solution.probabilities, descending=True)
    bitstrings_ = solution.bitstrings[sorted_indices][0:3, :]
    # Sort them by lexicographic order
    np_sorted_indices = np.lexsort(bitstrings_.numpy().T[::-1])
    bitstrings_ = bitstrings_[np_sorted_indices, :]

    torch.testing.assert_close(bitstrings_, bitstrings.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

    print(f"\n{analysis.to_dataframe([solution])}")

    # Remote solutions should be identical to local ones
    np.random.seed(seed)

    assert isinstance(results, Results)
    solver_remote = trivial_triangular_qubo(make_mock_connection(results, running_iterations=1))

    embedding = solver_remote._embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver_remote._drive(embedding)
    remote_job = solver_remote._submit(drive, embedding)

    if not wait:
        with pytest.raises(TimeoutError):
            remote_job.results(0)

    results_remote = remote_job.results()
    assert isinstance(results_remote, Results)
    check.equal(remote_job.get_status(), JobStatus.DONE)

    solution_remote = Solution.from_results(results, solver.instance)
    torch.testing.assert_close(solution_remote.bitstrings, solution.bitstrings)
    torch.testing.assert_close(solution_remote.counts, solution.counts)


def test_respects_total_bottom_detuning(caplog: pytest.LogCaptureFixture) -> None:
    n = 6
    Q = matrix.zeros(n)
    for i in range(n):
        Q[i, i] = -50.0 if i % 2 == 0 else 50.0
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = Q[j, i] = 1.0

    instance = Instance(Q)
    config = solvers.Config(solving=solvers.quantum.Config(drive_shaping=drive_shaping.Config(dmm=True)))

    with caplog.at_level(logging.INFO):
        solution = Solver(instance, config).solve()

    assert "DMM final detuning would exceed the device's total_bottom_detuning" in caplog.text

    assert solution.bitstrings.numel() > 0


def _triangular_register_qubo() -> np.ndarray:
    data_graph = DataGraph.triangular(4, 4, 1)
    np.random.seed(0)
    removed = np.random.choice(
        data_graph.number_of_nodes(), data_graph.number_of_nodes() - 8, replace=False
    )
    data_graph.remove_nodes_from(removed)
    coords = [data_graph.coords[n] for n in data_graph.nodes]
    dist_matrix = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        qubo = 1.0 / dist_matrix**6
    np.fill_diagonal(qubo, -0.5)
    return np.asarray(qubo, dtype=float)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("algorithm", ["greedy", "blade"])
def test_quantum_matches_classical_triangular(algorithm: str) -> None:
    qubo = _triangular_register_qubo()
    instance = Instance(matrix.tensor(qubo))

    quantum_config = solvers.Config(
        solving=solvers.quantum.Config(
            embedding=embedding.Config(algorithm=algorithm),
            drive_shaping=drive_shaping.Config(algorithm=Algorithm.PROPORTIONAL_DIAGONAL),
        ),
        do_preprocessing=False,
        do_postprocessing=False,
    )
    quantum_solution = Solver(instance, quantum_config).solve()
    quantum_solution._sort_by_cost()

    classical_config = solvers.Config(
        solving=solvers.classical.Config(max_bitstrings=4),
    )
    classical_solution = Solver(instance, classical_config).solve()
    classical_solution._sort_by_cost()

    check.almost_equal(
        quantum_solution.costs[0].item(),
        classical_solution.costs[0].item(),
        abs=1e-4,
    )

    assert quantum_solution.probabilities is not None
    check.greater_equal(quantum_solution.probabilities[0], 0.1)
