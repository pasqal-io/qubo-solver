from __future__ import annotations
from unittest.mock import patch, MagicMock

import pytest
import pytest_check as check
import torch


import pulser
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend

import qoolqit

from qubosolver import QUBOInstance, SolverConfig, EmbeddingConfig
from qubosolver import LocalEmulator as QoolqitLocalEmulator
from qubosolver.backends import _AutoLocalEmulatorBackend, LocalEmulator
from qubosolver.solver import QuboSolver


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
    backend = _AutoLocalEmulatorBackend(sequence)  # type: ignore[abstract]
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
        backend=QoolqitLocalEmulator(backend_type=_AutoLocalEmulatorBackend),  # type: ignore[type-abstract]
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
    check.is_instance(config.backend._backend_type, _AutoLocalEmulatorBackend)


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
