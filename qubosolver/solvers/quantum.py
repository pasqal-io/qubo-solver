from __future__ import annotations

import qoolqit
from qoolqit.execution.compilation_functions import CompilerProfile
from qoolqit.execution import job
from qubosolver.types import _protocols


def _quantum_program(
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    device: qoolqit.Device,
    *,
    compiler_profile: CompilerProfile = CompilerProfile.MAX_ENERGY,
) -> qoolqit.QuantumProgram:
    """Compile a :class:`qoolqit.QuantumProgram` for the target device.

    Args:
        device: Target quantum device specification.
        drive: Drive schedule containing the quantum operations.
        register: Register layout defining the qubit positions.
        compiler_profile: Compilation strategy. Defaults to ``MAX_ENERGY``.

    Returns:
        A compiled :class:`qoolqit.QuantumProgram`.
    """
    program = qoolqit.QuantumProgram(
        register=register,
        drive=drive,
    )
    max_duration_ratio = 0.99 if device.specs["max_duration"] is not None else None

    program.compile_to(
        device,
        profile=compiler_profile,
        device_max_duration_ratio=max_duration_ratio,
    )
    return program


def analog_quantum_sample(
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    backend: _protocols.Backend,
    device: qoolqit.Device,
    *,
    compiler_profile: CompilerProfile = CompilerProfile.MAX_ENERGY,
) -> job.Job:
    """Submit a quantum program for execution on the configured backend.

    Compiles the drive and register into a :class:`qoolqit.QuantumProgram`,
    then runs it on the provided backend.

    Args:
        backend: Execution backend (local emulator or remote QPU).
        device: Target quantum device specification.
        drive: Drive schedule for the quantum operations.
        register: Register layout defining the qubit positions.
        compiler_profile: Compilation strategy. Defaults to ``MAX_ENERGY``.

    Returns:
        A :class:`~qoolqit.execution.job.Job` handle for the submitted execution.
    """
    program = _quantum_program(register, drive, device, compiler_profile=compiler_profile)

    return backend.run(program)
