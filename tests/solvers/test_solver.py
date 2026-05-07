from __future__ import annotations

import numpy as np
import pytest

import io
import pytest_check as check
import torch
from unittest.mock import Mock

from qoolqit.devices import Device
from qoolqit.execution import JobStatus

from qubosolver.config import (
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.qubo_types import EmbedderType
from qubosolver.solver import (
    QUBOInstance,
    QuboSolver,
    QuboSolverClassical,
    QuboSolverQuantum,
    QUBOSolution,
)
from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.pipeline.basesolver import BaseSolver
from mock.connection import MockConnection

from pulser.backend.remote import (
    Results,
    RemoteConnection,
)
from emu_sv import SVBackend

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


@pytest.fixture
def implicit_default_qubo_solver_config(
    simple_qubo_instance: QUBOInstance,
) -> QuboSolver:
    default_solver = QuboSolver(simple_qubo_instance)
    return default_solver


def test_implicit_solver_config(
    implicit_default_qubo_solver_config: QuboSolver,
) -> None:
    assert isinstance(implicit_default_qubo_solver_config._solver, QuboSolverClassical)


def test_different_shots(simple_qubo_instance: QUBOInstance) -> None:
    from pulser_simulation import QutipBackendV2

    default_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500)
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500  # type: ignore[union-attr]

    lessshots_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=100)
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100  # type: ignore[union-attr]


@pytest.mark.priority(40)
@pytest.mark.flaky(max_runs=5)
def test_run_local_backends(
    simple_qubo_instance: QUBOInstance, local_backend: LocalEmulator
) -> None:
    solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True,
            backend=local_backend,
            embedding=EmbeddingConfig(embedding_method=EmbedderType.BLADE),
        ),
    )
    solutions = solver.solve()
    # theoretically -4.4000 can be found
    assert solutions.costs.min().item() <= -3.0


@pytest.mark.priority(150)
def test_solver_different_devices(
    request: pytest.FixtureRequest,
    qubo_for_testing_many_devices: QUBOInstance,
    local_device: Device,
    embedding_method: EmbedderType,
) -> None:

    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method="heuristic"),
        embedding=EmbeddingConfig(
            embedding_method=embedding_method,
            greedy_traps=qubo_for_testing_many_devices.size,
            min_distance=1.001,
        ),
        do_postprocessing=False,
        do_preprocessing=False,
        device=local_device,
        backend=LocalEmulator(backend_type=SVBackend),
    )
    solver = QuboSolver(qubo_for_testing_many_devices, config)
    solution = solver.solve()
    assert solution


def test_parse_results() -> None:
    # Mock results object
    mock_result = Mock()
    mock_result.final_bitstrings = {"001": 10, "110": 5, "010": 3}

    bitstrings, counts = BaseSolver.parse_results(mock_result)

    expected_bitstrings = torch.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    expected_counts = torch.tensor([10, 5, 3])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_empty_final_bitstrings() -> None:
    # Mock results with empty final_bitstrings
    mock_result = Mock()
    mock_result.final_bitstrings = {}

    bitstrings, counts = BaseSolver.parse_results(mock_result)

    check.equal(bitstrings.shape, (0, 0))
    check.equal(bitstrings.dtype, torch.int64)
    check.equal(counts.shape, (0,))
    check.equal(counts.dtype, torch.int64)


def test_parse_results_binary_string_conversion() -> None:
    # Mock results with binary string keys
    mock_result = Mock()
    mock_result.final_bitstrings = {"0101": 8, "1010": 12, "1111": 4}

    bitstrings, counts = BaseSolver.parse_results(mock_result)

    expected_bitstrings = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    expected_counts = torch.tensor([8, 12, 4])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_single_bitstring() -> None:
    # Mock results with single bitstring
    mock_result = Mock()
    mock_result.final_bitstrings = {"101": 25}

    bitstrings, counts = BaseSolver.parse_results(mock_result)

    expected_bitstrings = torch.tensor([[1, 0, 1]])
    expected_counts = torch.tensor([25])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_string_counts_to_integer_tensor() -> None:
    # Mock results with string count values
    mock_result = Mock()
    mock_result.final_bitstrings = {"101": "15", "010": "8", "111": "12"}

    bitstrings, counts = BaseSolver.parse_results(mock_result)

    expected_bitstrings = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    expected_counts = torch.tensor([15, 8, 12])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def trivial_triangular_qubo(connection: Optional[RemoteConnection] = None) -> QuboSolverQuantum:
    Q = 10.0 * np.array(
        [
            [-10.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    qubo = QUBOInstance(Q)

    config = SolverConfig(use_quantum=True, do_preprocessing=False)

    config.embedding = EmbeddingConfig(embedding_method="blade")
    num_shots = 100

    if connection is None:
        config.backend = LocalEmulator(num_shots=num_shots)
    else:
        config.backend = RemoteEmulator(connection=connection, num_shots=num_shots)

    solver = QuboSolverQuantum(qubo, config)
    solver._check_size_limit()

    return solver


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("wait", [True, False], ids=["wait", "dont_wait"])
def test_submit_integration(make_mock_connection: type[MockConnection], wait: bool) -> None:
    seed = 16844214
    np.random.seed(seed)

    solver = trivial_triangular_qubo()

    embedding = solver.embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver.drive(embedding)

    job = solver.submit(drive, embedding)
    results = job.results()
    bitstrings_local, counts_local = QuboSolverQuantum.parse_results(results)

    solution = QUBOSolution(
        bitstrings=bitstrings_local.float(),
        counts=counts_local,
        costs=torch.Tensor(),
        probabilities=None,
    )

    solution.costs = solution.compute_costs(solver.instance)
    solution.probabilities = solution.compute_probabilities()

    # Take the top 3 solutions with the highest probabilities
    sorted_indices = torch.argsort(solution.probabilities, descending=True)
    bitstrings = solution.bitstrings[sorted_indices].long()[0:3, :]
    # Sort them by lexicographic order
    np_sorted_indices = np.lexsort(bitstrings.numpy().T[::-1])
    bitstrings = bitstrings[np_sorted_indices, :]

    torch.testing.assert_close(bitstrings, torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

    solution.bitstrings = solution.bitstrings.int()
    analyzer = QUBOAnalyzer([solution])
    print(f"\n{analyzer.df}")

    # Remote solutions should be identical to local ones
    np.random.seed(seed)

    assert isinstance(results, Results)
    solver_remote = trivial_triangular_qubo(make_mock_connection(results, running_iterations=1))

    embedding = solver_remote.embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver_remote.drive(embedding)
    remote_job = solver_remote.submit(drive, embedding)

    if not wait:
        with pytest.raises(TimeoutError):
            remote_job.results(0)

    results_remote = remote_job.results()
    assert isinstance(results_remote, Results)
    check.equal(remote_job.get_status(), JobStatus.DONE)

    bitstrings_remote, counts_remote = QuboSolverQuantum.parse_results(results_remote)
    torch.testing.assert_close(bitstrings_remote, bitstrings_local)
    torch.testing.assert_close(counts_remote, counts_local)


@pytest.mark.parametrize("preprocessing", [True, False])
@pytest.mark.parametrize("postprocessing", [True, False])
def test_save_load_qubo_solver_quantum(
    preprocessing: bool,
    postprocessing: bool,
) -> None:

    Q = torch.tensor(
        [
            [0.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    expected_preprocessed_Q = torch.tensor(
        [
            [-10.0, 6.0],
            [6.0, -10.0],
        ]
    )
    qubo = QUBOInstance(Q)
    config = SolverConfig(do_preprocessing=preprocessing, do_postprocessing=postprocessing)
    solver = QuboSolverQuantum(qubo, config)
    solver._check_size_limit()
    solver.preprocess()

    if preprocessing:
        torch.testing.assert_close(solver.instance.coefficients, expected_preprocessed_Q)
    else:
        torch.testing.assert_close(solver.instance.coefficients, Q)
    torch.testing.assert_close(solver.fixtures.instance.coefficients, Q)

    # Save the solver
    file = io.BytesIO()
    QuboSolverQuantum.save(file, solver)

    # Load the solver
    file.seek(0)
    loaded_solver = QuboSolverQuantum.load(file)

    # Verify the loaded solver has the same properties
    # No need to have saved the preprocessed Q
    torch.testing.assert_close(loaded_solver.instance.coefficients, Q)
    torch.testing.assert_close(loaded_solver.fixtures.instance.coefficients, Q)
    check.equal(loaded_solver.config.do_preprocessing, solver.config.do_preprocessing)
    check.equal(loaded_solver.config.do_postprocessing, solver.config.do_postprocessing)
    check.equal(loaded_solver.fixtures.fixed_var_dict_list, solver.fixtures.fixed_var_dict_list)

    for method in [
        "solve",
        "embedding",
        "drive",
        "submit",
        "execute",
        "draw_sequence",
        "preprocess",
        "_trivial_solution",
    ]:
        with pytest.raises(
            AttributeError,
            match=f"'{method}' is disabled: this method is not supported for QuboSolverQuantum loaded from a file.",
        ):
            getattr(loaded_solver, method)()
