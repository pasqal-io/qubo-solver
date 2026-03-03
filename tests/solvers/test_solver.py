from __future__ import annotations

import pytest
import pytest_check as check
import torch
from unittest.mock import Mock

from qoolqit.devices import Device, DigitalAnalogDevice, AnalogDevice
from qubosolver.config import EmbeddingConfig, DriveShapingConfig, SolverConfig, LocalEmulator
from qubosolver.qubo_types import EmbedderType
from qubosolver.solver import QUBOInstance, QuboSolver, QuboSolverClassical
from qubosolver.pipeline.basesolver import BaseSolver


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
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, runs=500)
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500  # type: ignore[union-attr]

    lessshots_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, runs=100)
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100  # type: ignore[union-attr]


@pytest.mark.flaky(reruns=5)
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


def test_solver_different_devices(
    request: pytest.FixtureRequest,
    qubo_for_testing_many_devices: QUBOInstance,
    local_device: Device,
    embedding_method: EmbedderType,
) -> None:
    if (
        request.node.callspec.params["qubo_for_testing_many_devices"]
        == "qubo_instance_adiabatic_tutorial"
        and (
            type(request.node.callspec.params["local_device"])
            in (DigitalAnalogDevice, AnalogDevice)
            or request.node.callspec.params["local_device"].name == "FRESNEL"
        )
        and request.node.callspec.params["embedding_method"] == EmbedderType.BLADE
    ):
        pytest.skip(
            "The compilation of the sequence for this combination should be addressed in a new PR."
        )

    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method="adiabatic"),
        embedding=EmbeddingConfig(
            embedding_method=embedding_method, greedy_traps=qubo_for_testing_many_devices.size
        ),
        do_postprocessing=False,
        do_preprocessing=False,
        device=local_device,
    )
    solver = QuboSolver(qubo_for_testing_many_devices, config)
    solution = solver.solve()
    assert solution


def test_parse_results_remote_emulator() -> None:
    # Mock RemoteResults object
    mock_remote_results = Mock()
    mock_result_item = Mock()
    mock_result_item.bitstring_counts = {"001": 10, "110": 5, "010": 3}
    mock_remote_results.__getitem__ = Mock(return_value=mock_result_item)

    bitstrings, counts = BaseSolver.parse_results(mock_remote_results)

    expected_bitstrings = torch.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    expected_counts = torch.tensor([10, 5, 3])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_local_emulator() -> None:
    # Mock local emulator results (tuple format)
    mock_result = Mock()
    mock_result.final_bitstrings = {"001": 10, "110": 5, "101": 3}
    results = (None, mock_result)

    bitstrings, counts = BaseSolver.parse_results(results)

    expected_bitstrings = torch.tensor([[0, 0, 1], [1, 1, 0], [1, 0, 1]])
    expected_counts = torch.tensor([10, 5, 3])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_empty_remote_bitstring_counts() -> None:
    # Mock remote results with empty bitstring_counts
    mock_result = Mock()
    mock_result.bitstring_counts = {}
    mock_results = [mock_result]

    bitstrings, counts = BaseSolver.parse_results(mock_results)

    check.equal(bitstrings.shape, (0, 0))
    check.equal(bitstrings.dtype, torch.int64)
    check.equal(counts.shape, (0,))
    check.equal(counts.dtype, torch.int64)


def test_parse_results_empty_local_final_bitstrings() -> None:
    # Mock local emulator results with empty final_bitstrings
    mock_result = Mock()
    mock_result.final_bitstrings = {}
    results = (None, mock_result)

    bitstrings, counts = BaseSolver.parse_results(results)

    check.equal(bitstrings.shape, (0, 0))
    check.equal(bitstrings.dtype, torch.int64)
    check.equal(counts.shape, (0,))
    check.equal(counts.dtype, torch.int64)


def test_parse_results_binary_string_conversion() -> None:
    # Mock remote results with binary string keys
    mock_result = Mock()
    mock_result.bitstring_counts = {"0101": 8, "1010": 12, "1111": 4}
    mock_results = [mock_result]

    bitstrings, counts = BaseSolver.parse_results(mock_results)

    expected_bitstrings = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    expected_counts = torch.tensor([8, 12, 4])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_single_bitstring() -> None:
    # Mock remote results with single bitstring
    mock_result = Mock()
    mock_result.bitstring_counts = {"101": 25}
    mock_results = [mock_result]

    bitstrings, counts = BaseSolver.parse_results(mock_results)

    expected_bitstrings = torch.tensor([[1, 0, 1]])
    expected_counts = torch.tensor([25])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)


def test_parse_results_string_counts_to_integer_tensor() -> None:
    # Mock remote results with string count values
    mock_result = Mock()
    mock_result.bitstring_counts = {"101": "15", "010": "8", "111": "12"}
    mock_results = [mock_result]

    bitstrings, counts = BaseSolver.parse_results(mock_results)

    expected_bitstrings = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    expected_counts = torch.tensor([15, 8, 12])

    torch.testing.assert_close(bitstrings, expected_bitstrings)
    torch.testing.assert_close(counts, expected_counts)
