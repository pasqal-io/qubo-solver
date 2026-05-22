"""Local quantum backend implementation with automatic backend selection.

This module provides a wrapper class for local quantum emulators that automatically
selects the optimal backend implementation based on the quantum register size.

The automatic selection optimizes performance by choosing backends that are
most efficient for the given problem size:
- Small problems (< 20 qubits): QutipBackendV2
- Medium problems (20-29 qubits): SVBackend
- Large problems (≥ 30 qubits): MPSBackend

References:
- SVBackend performance benchmarks: https://pasqal-io.github.io/emulators/latest/emu_sv/benchmarks/performance/
- MPSBackend performance benchmarks: https://pasqal-io.github.io/emulators/latest/emu_mps/benchmarks/
- Backend selection methodology: https://arxiv.org/pdf/2510.09813
"""

from __future__ import annotations

from typing import Any, Type, cast

from pulser import Sequence as PulserSequence
from pulser.backend.abc import EmulatorBackend
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit.execution import LocalEmulator as QoolqitLocalEmulator

# Thresholds for automatic backend selection based on number of qubits
_MPS_THRESHOLD = 30  # Use MPS-based backends for problems with ≥30 qubits
_SV_THRESHOLD = 20  # Use state-vector backends for problems with ≥20 qubits


def _select_backend_type(n_quibts: int) -> Type[EmulatorBackend]:
    """Select the optimal backend class based on the number of qubits.

    Args:
        n_qubits (int): Number of qubits in the quantum register.

    Returns:
        Type[EmulatorBackend]: The selected backend class optimized for the given size.
    """
    if n_quibts >= _MPS_THRESHOLD:
        assert issubclass(MPSBackend, EmulatorBackend)
        return cast(Type[EmulatorBackend], MPSBackend)
    elif n_quibts >= _SV_THRESHOLD:
        assert issubclass(SVBackend, EmulatorBackend)
        return cast(Type[EmulatorBackend], SVBackend)
    else:
        return QutipBackendV2


class _AutoLocalEmulatorBackend(EmulatorBackend):
    """Factory class that automatically selects optimal emulator backends.

    This factory uses __new__ to return instances of different backend types
    based on quantum register size for optimal performance:
    - MPSBackend for large problems (≥30 qubits)
    - SVBackend for medium problems (20-29 qubits)
    - QutipBackendV2 for small problems (<20 qubits)

    Note: This class acts as a factory and never instantiates itself.
    The __new__ method directly returns instances of the selected backend type.
    Type checking is suppressed as this factory pattern confuses static analyzers.

    Required by qoolqit.LocalEmulator which expects backend_type to pass
    issubclass(backend_type, EmulatorBackend) checks.

    Args:
        sequence (PulserSequence): The pulse sequence to simulate.
        *args: Additional positional arguments passed to the selected backend.
        **kwargs: Additional keyword arguments passed to the selected backend.

    Returns:
        EmulatorBackend: An instance of the automatically selected backend
        (MPSBackend, SVBackend, or QutipBackendV2).
    """

    def __new__(cls, sequence: PulserSequence, *args: Any, **kwargs: Any) -> EmulatorBackend:  # type: ignore[misc]
        n_qubits = len(sequence.register.qubit_ids)
        return _select_backend_type(n_qubits)(sequence, *args, **kwargs)


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
