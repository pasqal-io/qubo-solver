from __future__ import annotations

from pulser import Sequence as PulserSequence
from pulser.backend.abc import EmulatorBackend
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit.execution import LocalEmulator as QoolqitLocalEmulator

_MPS_THRESHOLD = 30
_SV_THRESHOLD = 20

class _AutoLocalEmulatorBackend(EmulatorBackend):
    """Factory that selects a backend based on register size.

    Uses __new__ instead of a plain factory function because
    qoolqit.LocalEmulator requires backend_type to pass an
    issubclass(backend_type, EmulatorBackend) check.
    """

    def __new__(cls, sequence: PulserSequence, *args, **kwargs):
        n_qubits = len(sequence.register.qubit_ids)
        if n_qubits >= _MPS_THRESHOLD:
            backend = super().__new__(MPSBackend)
        elif n_qubits >= _SV_THRESHOLD:
            backend = super().__new__(SVBackend)
        else:
            backend = super().__new__(QutipBackendV2)
        backend.__init__(sequence, *args, **kwargs)
        return backend


class LocalEmulator(QoolqitLocalEmulator):

    def __init__(self, backend_type=_AutoLocalEmulatorBackend, **kwargs):
        super().__init__(backend_type=backend_type, **kwargs)
