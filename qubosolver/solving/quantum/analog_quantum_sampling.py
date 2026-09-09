"""Analog quantum sampling compilation.

The primary quantum solving primitive that compiles a pulse program for a target
device. Callers submit the compiled program themselves via ``backend.run(program)``
to sample bitstrings from the quantum state.
"""

from __future__ import annotations

import dataclasses

import qoolqit
from qoolqit.execution.compilation_functions import CompilerProfile


def compile(
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    device: qoolqit.Device,
    *,
    compiler_profile: CompilerProfile = CompilerProfile.MAX_ENERGY,
    default_sequence_duration: int | None = None,
) -> qoolqit.QuantumProgram:
    """Build and compile a [`qoolqit.QuantumProgram`][] for the target device.

    Args:
        register: Atom register defining qubit positions.
        drive: Pulse drive schedule encoding the quantum operations.
        device: Target quantum device that provides hardware constraints used
            during compilation.
        compiler_profile: Compilation strategy controlling how the pulse
            sequence is mapped to device constraints.
        default_sequence_duration: Fallback maximum sequence duration (ns)
            injected when `device` has no native `max_duration` cap.
            [`None`][] leaves the device unpatched.

    Returns:
        A compiled quantum program ready to be executed on a quantum backend
            (local/remote emulator, QPU).

    Example:
        ```python
        program = compile(register, drive, device)
        # Submit the program to a quantum backend
        job = backend.run(program)
        results = job.results()
        ```
    """
    if device.specs["max_duration"] is None and default_sequence_duration is not None:
        device_with_duration = dataclasses.replace(
            device._device,
            max_sequence_duration=default_sequence_duration,
        )
        device = qoolqit.Device(
            pulser_device=device_with_duration, default_converter=device.converter
        )

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
