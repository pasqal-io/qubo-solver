"""Analog quantum sampling compilation.

The primary quantum solving primitive that compiles a pulse program for a target
device. Callers submit the compiled program themselves via ``backend.run(program)``
to sample bitstrings from the quantum state.
"""

from __future__ import annotations

import dataclasses

import qoolqit
from qoolqit.execution.compilation_functions import CompilerProfile
from qubosolver.utils.quantum import _max_min_distance_ratio


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

    Raises:
        ValueError: If `register`'s max/min radial distance ratio exceeds what
            `device` allows, e.g. because it was embedded without device
            constraints in mind.

    Example:
        ```python
        program = compile(register, drive, device)
        # Submit the program to a quantum backend
        job = backend.run(program)
        results = job.results()
        ```
    """
    max_min_distance_ratio = register.max_radial_distance() / register.min_distance()
    device_max_min_distance_ratio = _max_min_distance_ratio(device)
    if max_min_distance_ratio > device_max_min_distance_ratio:
        raise ValueError(
            f"Register max/min distance ratio ({max_min_distance_ratio:.3g}) exceeds "
            f"the device's maximum allowed ratio ({device_max_min_distance_ratio:.3g}). "
            "This usually means the register was embedded without the target device's "
            "constraints in mind. Did you pass `device` to your embedding algorithm's config?"
        )

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
