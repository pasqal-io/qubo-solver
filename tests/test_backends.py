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
from pulser_pasqal.backends import (
    EmuFreeBackendV2,
    EmuSVBackend,
    EmuMPSBackend,
    RemoteEmulatorBackend,
)

import qoolqit

from qubosolver import QUBOInstance, SolverConfig, EmbeddingConfig
from qubosolver import LocalEmulator as QoolqitLocalEmulator
from qubosolver.backends import (
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    LocalEmulator,
    RemoteEmulator,
    _get_backend_type,
    _warn_suboptimal_backend,
)
from qubosolver.solver import QuboSolver
from mock.connection import MockConnection


def make_sequence(register: pulser.Register, device: qoolqit.Device) -> pulser.Sequence:
    sequence = pulser.Sequence(register, device._device)
    sequence.declare_channel("rydberg", "rydberg_global")
    sequence.add(
        pulser.Pulse.ConstantPulse(200, 1.0, 0.0, 0.0),
        "rydberg",
    )
    return sequence


def dummy_pulser_register(n: int) -> pulser.Register:
    qubits = {f"q{i}": (float(i), 0.0) for i in range(n)}
    return pulser.Register(qubits)


def mock_connection_and_results(size: int) -> tuple[MockConnection, pulser.backend.RemoteResults]:

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

    device = qoolqit.MockDevice()
    sequence = make_sequence(dummy_pulser_register(size), device)
    backend = AutoLocalEmulatorBackend(sequence)  # type: ignore[abstract]
    check.is_instance(backend, expected_type)


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (10, EmuFreeBackendV2),
        (20, EmuSVBackend),
        (30, EmuMPSBackend),
    ],
)
def test_auto_remote_emulator_backend(size: int, expected_type: type) -> None:

    device = qoolqit.MockDevice()
    sequence = make_sequence(dummy_pulser_register(size), device)
    backend = AutoRemoteEmulatorBackend(
        sequence, MagicMock(spec=pulser.backend.remote.RemoteConnection)
    )
    check.is_instance(backend, expected_type)


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (2, QutipBackendV2),
        (20, SVBackend),
        (30, MPSBackend),
    ],
)
def test_auto_local_emulator_backend_run(size: int, expected_type: type) -> None:

    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)
    config = SolverConfig(
        use_quantum=True,
        backend=QoolqitLocalEmulator(backend_type=AutoLocalEmulatorBackend),  # type: ignore[type-abstract]
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(
        expected_type, "run", return_value=MagicMock(spec=pulser.backend.Results)
    ) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (2, EmuFreeBackendV2),
        (20, EmuSVBackend),
        (30, EmuMPSBackend),
    ],
)
def test_auto_remote_emulator_backend_run(size: int, expected_type: type) -> None:

    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)

    # Mock connection for remote emulator
    mock_connection, mock_results = mock_connection_and_results(size)
    config = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(backend_type=AutoRemoteEmulatorBackend, connection=mock_connection),
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(expected_type, "run", return_value=mock_results) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (2, QutipBackendV2),
        (20, SVBackend),
        (30, MPSBackend),
    ],
)
def test_auto_local_emulator_run(size: int, expected_type: type) -> None:

    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)
    config = SolverConfig(
        use_quantum=True,
        backend=LocalEmulator(),
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(
        expected_type, "run", return_value=MagicMock(spec=pulser.backend.Results)
    ) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


def test_default_config_backend() -> None:
    config = SolverConfig(use_quantum=True)
    check.is_(config.backend._backend_type, AutoLocalEmulatorBackend)


def test_default_remote_emulator_backend() -> None:
    mock_connection = MagicMock(spec=pulser.backend.remote.RemoteConnection)
    emulator = RemoteEmulator(connection=mock_connection)
    check.is_(emulator._backend_type, EmuFreeBackendV2)


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (2, QutipBackendV2),
        (20, SVBackend),
        (30, MPSBackend),
    ],
)
def test_auto_local_emulator_run_with_default_config(size: int, expected_type: type) -> None:

    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)
    config = SolverConfig(
        use_quantum=True,
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(
        expected_type, "run", return_value=MagicMock(spec=pulser.backend.Results)
    ) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


@pytest.mark.parametrize(
    "size, expected_type",
    [
        (2, EmuFreeBackendV2),
        (20, EmuFreeBackendV2),
        (30, EmuFreeBackendV2),
    ],
)
def test_auto_remote_emulator_run_with_default_config(size: int, expected_type: type) -> None:

    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)

    mock_connection, mock_results = mock_connection_and_results(size)
    config = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(connection=mock_connection),  # Uses EmuFreeBackendV2 by default
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(expected_type, "run", return_value=mock_results) as mock_run:
        solver.solve()
        mock_run.assert_called_once()


def test_remote_emulator_warning() -> None:
    """Test that RemoteEmulator warns when using suboptimal backend."""
    size = 2
    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)
    mock_connection, mock_results = mock_connection_and_results(size)
    config = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(backend_type=EmuSVBackend, connection=mock_connection),
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )
    solver = QuboSolver(instance, config)

    with patch.object(EmuSVBackend, "run", return_value=mock_results) as mock_run:
        with pytest.warns(UserWarning, match="Consider using EmuFreeBackendV2"):
            solver.solve()
            mock_run.assert_called_once()


def test_local_emulator_warning() -> None:
    """Test that LocalEmulator warns when using suboptimal backend."""
    size = 2
    Q = torch.ones(size, size) + torch.diag(torch.full((size,), -3.0))
    instance = QUBOInstance(Q)
    config = SolverConfig(
        use_quantum=True,
        backend=LocalEmulator(backend_type=SVBackend),
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(embedding_method="blade", min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(
        SVBackend, "run", return_value=MagicMock(spec=pulser.backend.Results)
    ) as mock_run:
        with pytest.warns(UserWarning, match="Consider using QutipBackendV2"):
            solver.solve()
            mock_run.assert_called_once()


@pytest.mark.parametrize(
    "backend_id, remote, expected_type",
    [
        ("qutip", False, QutipBackendV2),
        ("qutip", True, EmuFreeBackendV2),
        ("emu_sv", False, SVBackend),
        ("emu_sv", True, EmuSVBackend),
        ("emu_mps", False, MPSBackend),
        ("emu_mps", True, EmuMPSBackend),
    ],
)
def test_get_backend_type(
    backend_id: Literal["qutip", "emu_sv", "emu_mps"], remote: bool, expected_type: type
) -> None:
    """Test that _get_backend_type returns the correct backend class."""
    backend_type = _get_backend_type(backend_id, remote)
    check.is_(backend_type, expected_type)
    if remote:
        assert issubclass(backend_type, RemoteEmulatorBackend)
    else:
        assert issubclass(backend_type, EmulatorBackend)


def test_get_backend_type_invalid_backend_id() -> None:
    """Test that _get_backend_type raises ValueError for invalid backend_id."""
    with pytest.raises(ValueError, match="not recognized"):
        _get_backend_type("invalid", False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "backend_type, n_qubits, remote, should_warn",
    [
        # Auto backends should never warn
        (AutoLocalEmulatorBackend, 10, False, False),
        (AutoLocalEmulatorBackend, 20, False, False),
        (AutoRemoteEmulatorBackend, 10, True, False),
        (AutoRemoteEmulatorBackend, 20, True, False),

        # Optimal backends should not warn
        (QutipBackendV2, 10, False, False),  # Optimal for <15 qubits local
        (EmuFreeBackendV2, 10, True, False),  # Optimal for <15 qubits remote
        (SVBackend, 20, False, False),  # Optimal for 15-25 qubits local
        (EmuSVBackend, 20, True, False),  # Optimal for 15-25 qubits remote
        (MPSBackend, 30, False, False),  # Optimal for ≥26 qubits local
        (EmuMPSBackend, 30, True, False),  # Optimal for ≥26 qubits remote

        # Suboptimal backends should warn
        (SVBackend, 10, False, True),  # SVBackend for small problem
        (MPSBackend, 10, False, True),  # MPSBackend for small problem
        (QutipBackendV2, 20, False, True),  # QutipBackendV2 for medium problem
        (EmuSVBackend, 10, True, True),  # EmuSVBackend for small problem
        (EmuMPSBackend, 10, True, True),  # EmuMPSBackend for small problem
        (EmuFreeBackendV2, 20, True, True),  # EmuFreeBackendV2 for medium problem
    ],
)
def test_warn_suboptimal_backend(backend_type: type, n_qubits: int, remote: bool, should_warn: bool) -> None:
    """Test that _warn_suboptimal_backend warns appropriately based on backend optimality."""
    if should_warn:
        with pytest.warns(UserWarning, match="Consider using"):
            _warn_suboptimal_backend(backend_type, n_qubits, remote)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            _warn_suboptimal_backend(backend_type, n_qubits, remote)  # Should not raise


def test_warn_suboptimal_backend_remote_message() -> None:
    """Test that remote backend warnings include fee notice."""
    with pytest.warns(UserWarning, match="Note: Fees may apply for remote execution"):
        _warn_suboptimal_backend(EmuSVBackend, 10, True)


def test_warn_suboptimal_backend_local_no_fee_message() -> None:
    """Test that local backend warnings do not include fee notice."""
    with pytest.warns(UserWarning) as warning_list:
        _warn_suboptimal_backend(SVBackend, 10, False)

    # Check that no warning mentions fees
    for warning in warning_list:
        assert "Fees may apply" not in str(warning.message)


def test_warn_suboptimal_backend_message_content() -> None:
    """Test the specific content of warning messages."""
    with pytest.warns(UserWarning, match=r"Using SVBackend for 10 qubits\. Consider using QutipBackendV2"):
        _warn_suboptimal_backend(SVBackend, 10, False)
