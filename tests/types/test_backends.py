from __future__ import annotations
from unittest.mock import patch, MagicMock

import pytest
import pytest_check as check
import torch
from typing import Literal
import warnings


import pulser
from pulser_simulation import QutipBackendV2
from pulser.backend.abc import EmulatorBackend
from emu_sv import SVBackend
from emu_mps import MPSBackend
from pasqal_cloud.backends import (
    RemoteEmuFreeBackend,
    RemoteSVBackend,
    RemoteMPSBackend,
    RemoteEmulatorBackend,
)

import qoolqit

from qubosolver import (
    Instance,
    solvers,
    embedding,
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    LocalEmulator,
    RemoteEmulator,
    Solver,
    matrix,
)
from qubosolver.types.backends import (
    _get_backend_type,
    _warn_suboptimal_backend,
)
from mock.connection import MockConnection


def make_sequence(register: pulser.Register, device: qoolqit.Device) -> pulser.Sequence:
    """Create a simple pulse sequence for testing purposes."""
    sequence = pulser.Sequence(register, device._device)
    sequence.declare_channel("rydberg", "rydberg_global")
    sequence.add(
        pulser.Pulse.ConstantPulse(200, 1.0, 0.0, 0.0),
        "rydberg",
    )
    return sequence


def dummy_pulser_register(n: int) -> pulser.Register:
    """Create a dummy pulser register with n qubits arranged in a line."""
    qubits = {f"q{i}": (float(i), 0.0) for i in range(n)}
    return pulser.Register(qubits)


def mock_connection_and_results() -> tuple[MockConnection, pulser.backend.RemoteResults]:
    """Create a mock connection and results for testing remote emulators."""
    mock_results = MagicMock(spec=pulser.backend.Results)
    mock_connection = MockConnection(mock_results)

    device = qoolqit.MockDevice()
    sequence = make_sequence(dummy_pulser_register(2), device)
    remote_results = mock_connection.submit(sequence)

    return mock_connection, remote_results


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (10, QutipBackendV2),
        (20, SVBackend),
        (30, MPSBackend),
    ],
)
def test_auto_local_emulator_backend(size: int, expected_type: type) -> None:
    """Test that AutoLocalEmulatorBackend selects the correct backend type based on problem size."""
    device = qoolqit.MockDevice()
    sequence = make_sequence(dummy_pulser_register(size), device)
    backend = AutoLocalEmulatorBackend(sequence)  # type: ignore[abstract]
    check.is_instance(backend, expected_type)


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (10, RemoteEmuFreeBackend),
        (20, RemoteSVBackend),
        (30, RemoteMPSBackend),
    ],
)
def test_auto_remote_emulator_backend(size: int, expected_type: type) -> None:
    """Test that AutoRemoteEmulatorBackend selects the correct backend type based on problem size."""
    device = qoolqit.MockDevice()
    sequence = make_sequence(dummy_pulser_register(size), device)
    backend = AutoRemoteEmulatorBackend(
        sequence, MagicMock(spec=pulser.backend.remote.RemoteConnection)
    )
    check.is_instance(backend, expected_type)


def attach_bitstring(
    results: pulser.backend.Results | pulser.backend.RemoteResults, size: int
) -> None:
    r = {
        f"{'0' * size}": 10,
    }
    if isinstance(results, pulser.backend.Results):
        results.final_bitstrings = r  # type: ignore[misc]
    else:
        results._results_seq[0].final_bitstrings = r


@pytest.fixture
def backend_and_results(
    request: pytest.FixtureRequest,
) -> (
    tuple[LocalEmulator, pulser.backend.Results]
    | tuple[RemoteEmulator, pulser.backend.RemoteResults]
):
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture
def local_default_backend() -> tuple[LocalEmulator, pulser.backend.Results]:
    results = MagicMock(spec=pulser.backend.Results)
    return LocalEmulator(), results


@pytest.fixture
def local_auto_backend() -> tuple[LocalEmulator, pulser.backend.Results]:
    results = MagicMock(spec=pulser.backend.Results)
    return LocalEmulator(backend_type=AutoLocalEmulatorBackend), results  # type: ignore[type-abstract]


@pytest.fixture
def local_default_config() -> tuple[LocalEmulator, pulser.backend.Results]:
    results = MagicMock(spec=pulser.backend.Results)
    backend = solvers.Config().quantum.backend
    assert isinstance(backend, LocalEmulator)
    return backend, results


@pytest.fixture
def remote_default_backend() -> tuple[RemoteEmulator, pulser.backend.RemoteResults]:
    mock_connection, results = mock_connection_and_results()
    return RemoteEmulator(connection=mock_connection), results


@pytest.fixture
def remote_auto_backend() -> tuple[RemoteEmulator, pulser.backend.RemoteResults]:
    mock_connection, results = mock_connection_and_results()
    return (
        RemoteEmulator(backend_type=AutoRemoteEmulatorBackend, connection=mock_connection),
        results,
    )


@pytest.mark.priority(120)
@pytest.mark.parametrize(
    "size, backend_and_results, expected_type",
    [
        # Auto backend tests - local
        (2, "local_auto_backend", QutipBackendV2),
        (20, "local_auto_backend", SVBackend),
        (30, "local_auto_backend", MPSBackend),
        # Auto backend tests - remote
        (2, "remote_auto_backend", RemoteEmuFreeBackend),
        (20, "remote_auto_backend", RemoteSVBackend),
        (30, "remote_auto_backend", RemoteMPSBackend),
        # Default backend tests - local
        (2, "local_default_backend", QutipBackendV2),
        (20, "local_default_backend", SVBackend),
        (30, "local_default_backend", MPSBackend),
        # Default backend tests - remote (always RemoteEmuFreeBackend)
        (2, "remote_default_backend", RemoteEmuFreeBackend),
        (20, "remote_default_backend", RemoteEmuFreeBackend),
        (30, "remote_default_backend", RemoteEmuFreeBackend),
        # Default config tests - local
        (2, "local_default_config", QutipBackendV2),
        (20, "local_default_config", SVBackend),
        (30, "local_default_config", MPSBackend),
    ],
    indirect=("backend_and_results",),
)
def test_emulator_backend_selection(
    size: int,
    backend_and_results: tuple,
    expected_type: type,
) -> None:
    """Test that emulators select the correct backend based on problem size and configuration."""
    Q = matrix.from_torch(torch.ones(size, size) + torch.diag(torch.full((size,), -3.0)))
    instance = Instance(Q)

    backend, results = backend_and_results
    attach_bitstring(results, size)

    solver_config = solvers.Config(
        solving=solvers.QuantumConfig(
            backend=backend,
            embedding=embedding.Config(algorithm="blade"),
        ),
        activate_trivial_solutions=False,
    )

    solver = Solver(instance, solver_config)
    with patch.object(expected_type, "run", return_value=results) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


def test_default_config_backend() -> None:
    """Test that default solvers.Config uses AutoLocalEmulatorBackend."""
    config = solvers.Config()
    assert isinstance(config.solving, solvers.QuantumConfig)
    check.is_(config.solving.backend._backend_type, AutoLocalEmulatorBackend)


def test_default_remote_emulator_backend() -> None:
    """Test that default RemoteEmulator uses RemoteEmuFreeBackend."""
    mock_connection = MagicMock(spec=pulser.backend.remote.RemoteConnection)
    emulator = RemoteEmulator(connection=mock_connection)
    check.is_(emulator._backend_type, RemoteEmuFreeBackend)


def test_remote_emulator_warning() -> None:
    """Test that RemoteEmulator warns when using suboptimal backend."""
    size = 2
    Q = matrix.from_torch(torch.ones(size, size) + torch.diag(torch.full((size,), -3.0)))
    instance = Instance(Q)
    mock_connection, mock_results = mock_connection_and_results()
    attach_bitstring(mock_results, size)
    config = solvers.Config(
        solving=solvers.QuantumConfig(
            backend=RemoteEmulator(backend_type=RemoteSVBackend, connection=mock_connection),
            embedding=embedding.Config(algorithm="blade"),
        ),
        activate_trivial_solutions=False,
    )
    solver = Solver(instance, config)

    with patch.object(RemoteSVBackend, "run", return_value=mock_results) as mock_run:
        with pytest.warns(UserWarning, match="Consider using RemoteEmuFreeBackend"):
            solver.solve()
            mock_run.assert_called_once()


def test_local_emulator_warning() -> None:
    """Test that LocalEmulator warns when using suboptimal backend."""
    size = 2
    Q = matrix.from_torch(torch.ones(size, size) + torch.diag(torch.full((size,), -3.0)))
    instance = Instance(Q)
    config = solvers.Config(
        solving=solvers.QuantumConfig(
            backend=LocalEmulator(backend_type=SVBackend),
            embedding=embedding.Config(algorithm="blade"),
        ),
        activate_trivial_solutions=False,
    )

    solver = Solver(instance, config)
    results = MagicMock(spec=pulser.backend.Results)
    attach_bitstring(results, size)
    with patch.object(SVBackend, "run", return_value=results) as mock_run:
        with pytest.warns(UserWarning, match="Consider using QutipBackendV2"):
            solver.solve()
            mock_run.assert_called_once()


@pytest.mark.parametrize(
    "backend_id, expected_type",
    [
        ("qutip", QutipBackendV2),
        ("emu_sv", SVBackend),
        ("emu_mps", MPSBackend),
    ],
)
def test_get_local_backend_type(
    backend_id: Literal["qutip", "emu_sv", "emu_mps"], expected_type: type
) -> None:
    """Test that _get_backend_type returns the correct local backend class."""
    backend_type = _get_backend_type(backend_id, False)
    check.is_(backend_type, expected_type)
    assert issubclass(backend_type, EmulatorBackend)


@pytest.mark.parametrize(
    "backend_id, expected_type",
    [
        ("qutip", RemoteEmuFreeBackend),
        ("emu_sv", RemoteSVBackend),
        ("emu_mps", RemoteMPSBackend),
    ],
)
def test_get_remote_backend_type(
    backend_id: Literal["qutip", "emu_sv", "emu_mps"], expected_type: type
) -> None:
    """Test that _get_backend_type returns the correct remote backend class."""
    backend_type = _get_backend_type(backend_id, True)
    check.is_(backend_type, expected_type)
    assert issubclass(backend_type, RemoteEmulatorBackend)


def test_get_backend_type_invalid_backend_id() -> None:
    """Test that _get_backend_type raises ValueError for invalid backend_id."""
    with pytest.raises(ValueError, match="not recognized"):
        _get_backend_type("invalid", False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "backend_type, n_qubits",
    [
        # Suboptimal backends should warn
        (SVBackend, 10),  # SVBackend for small problem
        (MPSBackend, 10),  # MPSBackend for small problem
        (QutipBackendV2, 20),  # QutipBackendV2 for medium problem
        (RemoteSVBackend, 10),  # RemoteSVBackend for small problem
        (RemoteMPSBackend, 10),  # RemoteMPSBackend for small problem
        (RemoteEmuFreeBackend, 20),  # RemoteEmuFreeBackend for medium problem
    ],
)
def test_warn_suboptimal_backend(
    backend_type: type,
    n_qubits: int,
) -> None:
    """Test that _warn_suboptimal_backend warns for suboptimal backend choices."""
    with pytest.warns(UserWarning, match="Consider using"):
        _warn_suboptimal_backend(backend_type, n_qubits)


@pytest.mark.parametrize(
    "backend_type, n_qubits",
    [
        # Auto backends should never warn
        (AutoLocalEmulatorBackend, 10),
        (AutoLocalEmulatorBackend, 20),
        (AutoRemoteEmulatorBackend, 10),
        (AutoRemoteEmulatorBackend, 20),
        # Optimal backends should not warn
        (QutipBackendV2, 10),  # Optimal for <15 qubits local
        (RemoteEmuFreeBackend, 10),  # Optimal for <15 qubits remote
        (SVBackend, 20),  # Optimal for 15-25 qubits local
        (RemoteSVBackend, 20),  # Optimal for 15-25 qubits remote
        (MPSBackend, 30),  # Optimal for ≥26 qubits local
        (RemoteMPSBackend, 30),  # Optimal for ≥26 qubits remote
    ],
)
def test_dont_warn_optimal_backend(
    backend_type: type,
    n_qubits: int,
) -> None:
    """Test that _warn_suboptimal_backend does not warn for optimal backend choices."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        _warn_suboptimal_backend(backend_type, n_qubits)  # Should not raise


def test_warn_suboptimal_backend_remote_message() -> None:
    """Test that remote backend warnings include fee notice."""
    with pytest.warns(UserWarning, match="Note: Fees may apply for remote execution"):
        _warn_suboptimal_backend(RemoteSVBackend, 10)


def test_warn_suboptimal_backend_local_no_fee_message() -> None:
    """Test that local backend warnings do not include fee notice."""
    with pytest.warns(UserWarning) as warning_list:
        _warn_suboptimal_backend(SVBackend, 10)

    # Check that no warning mentions fees
    for warning in warning_list:
        assert "Fees may apply" not in str(warning.message)


def test_warn_suboptimal_backend_message_content() -> None:
    """Test the specific content of warning messages."""
    with pytest.warns(
        UserWarning, match=r"Using SVBackend for 10 qubits\. Consider using QutipBackendV2"
    ):
        _warn_suboptimal_backend(SVBackend, 10)
