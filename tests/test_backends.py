from __future__ import annotations
from unittest.mock import patch, MagicMock

import pytest_check as check
import torch


import pulser
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend

import qoolqit

from qubosolver import QUBOInstance, SolverConfig, LocalEmulator, EmbeddingConfig
from qubosolver.backends import AutoLocalEmulatorBackend
from qubosolver.solver import QuboSolver


def make_sequence(register: pulser.Register, device: qoolqit.Device) -> pulser.Sequence:
    sequence = pulser.Sequence(register, device._device)
    sequence.declare_channel("rydberg", "rydberg_global")
    sequence.add(
        pulser.Pulse.ConstantPulse(200, 1.0, 0.0, 0.0),
        "rydberg",
    )
    return sequence

def test_auto_local_emulator_backend() -> None:

    def dummy_pulser_register(size: int) -> pulser.Register:
        qubits = { f"q{i}": (float(i), 0.) for i in range(size) }
        return pulser.Register(qubits)

    device = qoolqit.MockDevice()

    backend_15 = AutoLocalEmulatorBackend(make_sequence(dummy_pulser_register(15), device))
    check.is_instance(backend_15, QutipBackendV2)

    backend_25 = AutoLocalEmulatorBackend(make_sequence(dummy_pulser_register(25), device))
    check.is_instance(backend_25, SVBackend)

    backend_35 = AutoLocalEmulatorBackend(make_sequence(dummy_pulser_register(35), device))
    check.is_instance(backend_35, MPSBackend)


def test_auto_local_emulator_backend_run() -> None:

    Q = torch.Tensor([[-1, 2], [2, -1]])
    instance = QUBOInstance(Q)
    config = SolverConfig(
        use_quantum=True,
        backend=LocalEmulator(backend_type=AutoLocalEmulatorBackend),
        activate_trivial_solutions=False,
        embedding=EmbeddingConfig(min_distance=1.001),
    )

    solver = QuboSolver(instance, config)
    with patch.object(QutipBackendV2, "run", return_value=MagicMock(spec=pulser.backend.Results)) as mock_run:
        solver.solve()
        mock_run.assert_called_once()



