from __future__ import annotations

from pulser import Sequence as PulserSequence
from pulser.backend.abc import EmulatorBackend
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit.execution import LocalEmulator as _LocalEmulator

_MPS_THRESHOLD = 30
_SV_THRESHOLD = 20

class _AutoLocalEmulatorBackend(EmulatorBackend):
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


class LocalEmulator(_LocalEmulator):
    def __init__(self, **kwargs):
        kwargs.setdefault("backend_type", _AutoLocalEmulatorBackend)
        super().__init__(**kwargs)
