"""Local quantum backend implementation with automatic backend selection.

This module provides a wrapper class for local quantum emulators that automatically
selects the optimal backend implementation based on the quantum register size.

The automatic selection optimizes performance by choosing backends that are
most efficient for the given problem size:
- Small problems (< 20 qubits): QutipBackendV2
- Medium problems (20-29 qubits): SVBackend
- Large problems (≥ 30 qubits): MPSBackend
"""

from __future__ import annotations

from pulser import Sequence as PulserSequence
from pulser.backend.abc import EmulatorBackend
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit.execution import LocalEmulator as QoolqitLocalEmulator

# Thresholds for automatic backend selection based on number of qubits
_MPS_THRESHOLD = 30  # Use MPS-based backends for problems with ≥30 qubits
_SV_THRESHOLD = 20   # Use state-vector backends for problems with ≥20 qubits

class _AutoLocalEmulatorBackend(EmulatorBackend):
    """Factory that selects a local emulator backend based on register size.

    This factory automatically chooses the most efficient local backend:
    - MPSBackend for large problems (≥30 qubits)
    - SVBackend for medium problems (20-29 qubits)
    - QutipBackendV2 for small problems (<20 qubits)

    Uses __new__ instead of a plain factory function because
    qoolqit.LocalEmulator requires backend_type to pass an
    issubclass(backend_type, EmulatorBackend) check.

    Args:
        sequence (PulserSequence): The pulse sequence to simulate.
        *args: Additional positional arguments passed to the backend.
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        EmulatorBackend: The selected backend instance.
    """

    def __new__(cls, sequence: PulserSequence, *args: Any, **kwargs: Any) -> EmulatorBackend:
        n_qubits = len(sequence.register.qubit_ids)
        if n_qubits >= _MPS_THRESHOLD:
            return MPSBackend(sequence, *args, **kwargs)
        elif n_qubits >= _SV_THRESHOLD:
            return SVBackend(sequence, *args, **kwargs)
        else:
            return QutipBackendV2(sequence, *args, **kwargs)


class LocalEmulator(QoolqitLocalEmulator):
    """Local quantum emulator with automatic backend selection.

    This class wraps qoolqit.LocalEmulator and automatically selects
    the optimal local backend based on the quantum register size.
    It provides the same interface as the base LocalEmulator but with
    improved performance through intelligent backend selection.

    Args:
        backend_type (type, optional): Backend type to use. Defaults to
            _AutoLocalEmulatorBackend for automatic selection.
        **kwargs: Additional keyword arguments passed to the base LocalEmulator.

    Example:
        >>> from qubosolver.backends import LocalEmulator
        >>> emulator = LocalEmulator(num_shots=1000)
        >>> # Backend will be automatically selected based on problem size
    """

    def __init__(
        self, backend_type: Type[EmulatorBackend] = _AutoLocalEmulatorBackend, **kwargs: Any
    ) -> None:
        super().__init__(backend_type=backend_type, **kwargs)
