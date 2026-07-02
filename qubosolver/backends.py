"""Local and remote quantum backend implementation with automatic backend selection.

This module provides wrapper classes for quantum emulators that automatically
select the optimal backend implementation based on the quantum register size.

The automatic selection optimizes performance by choosing backends that are
most efficient for the given problem size:
- Small problems (< 15 qubits): QutipBackendV2 (local) / RemoteEmuFreeBackend (remote)
- Medium problems (15-25 qubits): SVBackend (local) / RemoteSVBackend (remote)
- Large problems (≥ 26 qubits): MPSBackend (local) / RemoteMPSBackend (remote)

References:
- SVBackend performance benchmarks: https://pasqal-io.github.io/emulators/latest/emu_sv/benchmarks/performance/
- MPSBackend performance benchmarks: https://pasqal-io.github.io/emulators/latest/emu_mps/benchmarks/
- Backend selection methodology: https://arxiv.org/pdf/2510.09813
"""

from __future__ import annotations

import warnings
from typing import Any, Type, cast, Literal

from pulser import Sequence as PulserSequence
from pulser.backend.abc import EmulatorBackend
from pulser_simulation import QutipBackendV2
from pasqal_cloud.backends import (
    RemoteEmulatorBackend,
    RemoteEmuFreeBackend,
    RemoteSVBackend,
    RemoteMPSBackend,
)
from emu_sv import SVBackend
from emu_mps import MPSBackend
import qoolqit
from qoolqit.execution import (
    LocalEmulator as QoolqitLocalEmulator,
    RemoteEmulator as QoolqitRemoteEmulator,
)

# Thresholds for automatic backend selection based on number of qubits
_MPS_THRESHOLD = 26  # Use MPS-based backends for problems with ≥26 qubits
_SV_THRESHOLD = 15  # Use state-vector backends for problems with ≥15 qubits


def _get_backend_type(
    backend_id: Literal["qutip", "emu_sv", "emu_mps"], remote: bool
) -> Type[EmulatorBackend] | Type[RemoteEmulatorBackend]:
    """Get the backend type for a given backend ID and execution mode.

    Args:
        backend_id (Literal["qutip", "emu_sv", "emu_mps"]): Backend identifier.
        remote (bool): Whether to get a remote or local backend type.

    Returns:
        Type[EmulatorBackend] | Type[RemoteEmulatorBackend]: The backend class for the specified ID and execution mode.

    Raises:
        ValueError: If the backend_id is not recognized.
    """
    match (backend_id, remote):
        case ("qutip", True):
            return RemoteEmuFreeBackend
        case ("qutip", False):
            return QutipBackendV2
        case ("emu_sv", True):
            return RemoteSVBackend
        case ("emu_sv", False):
            return cast(Type[EmulatorBackend], SVBackend)
        case ("emu_mps", True):
            return RemoteMPSBackend
        case ("emu_mps", False):
            return cast(Type[EmulatorBackend], MPSBackend)
        case _:
            raise ValueError(f"Backend ID '{backend_id}' is not recognized")


def _select_backend_type(
    n_qubits: int, remote: bool
) -> Type[EmulatorBackend] | Type[RemoteEmulatorBackend]:
    """Select the appropriate backend class based on the number of qubits.

    Args:
        n_qubits (int): Number of qubits in the quantum register.
        remote (bool): Whether to select a remote or local backend.

    Returns:
        Type[EmulatorBackend] | Type[RemoteEmulatorBackend]: The selected backend class
        appropriate for the given problem size. QutipBackendV2/RemoteEmuFreeBackend become
        intractable beyond 15 qubits.
    """
    if n_qubits >= _MPS_THRESHOLD:
        return _get_backend_type("emu_mps", remote)
    elif n_qubits >= _SV_THRESHOLD:
        return _get_backend_type("emu_sv", remote)
    else:
        return _get_backend_type("qutip", remote)


class AutoLocalEmulatorBackend(EmulatorBackend):
    """Factory class that automatically selects optimal emulator backends.

    This factory uses __new__ to return instances of different backend types
    based on quantum register size for optimal performance:
    - MPSBackend for large problems (≥26 qubits)
    - SVBackend for medium problems (15-25 qubits)
    - QutipBackendV2 for small problems (<15 qubits)

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
        return _select_backend_type(n_qubits, False)(sequence, *args, **kwargs)


class AutoRemoteEmulatorBackend(RemoteEmulatorBackend):
    """Factory class that automatically selects optimal remote emulator backends.

    This factory uses __new__ to return instances of different remote backend types
    based on quantum register size for optimal performance:
    - RemoteMPSBackend for large problems (≥26 qubits)
    - RemoteSVBackend for medium problems (15-25 qubits)
    - RemoteEmuFreeBackend for small problems (<15 qubits)

    Note: This class acts as a factory and never instantiates itself.
    The __new__ method directly returns instances of the selected remote backend type.

    Args:
        sequence (PulserSequence): The pulse sequence to simulate.
        *args: Additional positional arguments passed to the selected backend.
        **kwargs: Additional keyword arguments passed to the selected backend.

    Returns:
        RemoteEmulatorBackend: An instance of the automatically selected remote backend
        (RemoteMPSBackend, RemoteSVBackend, or RemoteEmuFreeBackend).
    """

    def __new__(cls, sequence: PulserSequence, *args: Any, **kwargs: Any) -> RemoteEmulatorBackend:  # type: ignore[misc]
        n_qubits = len(sequence.register.qubit_ids)
        backend = _select_backend_type(n_qubits, True)(sequence, *args, **kwargs)
        assert isinstance(backend, RemoteEmulatorBackend)  # nosec B101
        return backend


def _warn_suboptimal_backend(
    backend_type: Type[EmulatorBackend] | Type[RemoteEmulatorBackend],
    n_qubits: int,
) -> None:
    """Warn if using a suboptimal backend for the given problem size.

    Args:
        backend_type: The currently selected backend type
        n_qubits: Number of qubits in the quantum program
    """
    if backend_type in [AutoLocalEmulatorBackend, AutoRemoteEmulatorBackend]:
        return

    remote = issubclass(backend_type, RemoteEmulatorBackend)

    optimal_backend_type = _select_backend_type(n_qubits, remote)

    if backend_type is optimal_backend_type:
        return

    warning_msg = (
        f"Using {backend_type.__name__} for {n_qubits} qubits. "
        f"Consider using {optimal_backend_type.__name__} which is recommended "
        f"for this problem size."
    )

    if remote:
        warning_msg += " Note: Fees may apply for remote execution."

    warnings.warn(warning_msg, UserWarning, stacklevel=2)


class LocalEmulator(QoolqitLocalEmulator):
    """Local quantum emulator with automatic backend selection.

    This class wraps qoolqit.LocalEmulator and automatically selects
    the optimal local backend based on the quantum register size.
    It provides the same interface as the base LocalEmulator but with
    improved performance through intelligent backend selection.

    The optimal backend selection follows these guidelines:
    - Small problems (< 15 qubits): QutipBackendV2
    - Medium problems (15-25 qubits): SVBackend
    - Large problems (≥ 26 qubits): MPSBackend

    Args:
        backend_type (type, optional): Backend type to use. Defaults to
            AutoLocalEmulatorBackend for automatic selection.
        **kwargs: Additional keyword arguments passed to the base LocalEmulator.

    Example:
        >>> from qubosolver.backends import LocalEmulator
        >>> emulator = LocalEmulator(num_shots=1000)
        >>> # Backend will be automatically selected based on problem size
    """

    def __init__(
        self, backend_type: Type[EmulatorBackend] = AutoLocalEmulatorBackend, **kwargs: Any
    ) -> None:
        super().__init__(backend_type=backend_type, **kwargs)

    def run(self, program: qoolqit.QuantumProgram, *args: Any, **kwargs: Any) -> Any:
        """Run the quantum program with backend tractability warning.

        Args:
            program: The quantum program to execute
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The execution results from the local backend
        """
        _warn_suboptimal_backend(self._backend_type, program.register.n_qubits)
        return super().run(program, *args, **kwargs)


class RemoteEmulator(QoolqitRemoteEmulator):
    """Remote quantum emulator with automatic backend selection.

    This class wraps qoolqit.RemoteEmulator and provides backend selection
    recommendations based on quantum register size and tractability constraints.

    Backend selection guidelines based on computational tractability:
    - Small problems (< 15 qubits): RemoteEmuFreeBackend (default)
    - Medium problems (15-25 qubits): RemoteSVBackend
    - Large problems (≥ 26 qubits): RemoteMPSBackend

    Note: RemoteEmuFreeBackend becomes intractable beyond ~15 qubits, similar to its
    local counterpart QutipBackendV2. For larger problems, RemoteSVBackend and
    RemoteMPSBackend are necessary. Fees may apply for remote execution.

    Args:
        backend_type (type, optional): Backend type to use. Defaults to
            RemoteEmuFreeBackend.
        **kwargs: Additional keyword arguments passed to the base RemoteEmulator.

    Example:
        >>> from qubosolver.backends import RemoteEmulator
        >>> from pasqal_cloud import PasqalCloudConnection
        >>> connection = PasqalCloudConnection(username="user", password="pass", project_id="project")
        >>> emulator = RemoteEmulator(connection=connection, num_shots=1000)
        >>> # Uses RemoteEmuFreeBackend by default
    """

    def __init__(
        self, backend_type: Type[RemoteEmulatorBackend] = RemoteEmuFreeBackend, **kwargs: Any
    ) -> None:
        super().__init__(backend_type=backend_type, **kwargs)

    def run(self, program: qoolqit.QuantumProgram, *args: Any, **kwargs: Any) -> Any:
        """Run the quantum program with backend tractability warning.

        Args:
            program: The quantum program to execute
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The execution results from the remote backend
        """
        _warn_suboptimal_backend(self._backend_type, program.register.n_qubits)
        return super().run(program, *args, **kwargs)
