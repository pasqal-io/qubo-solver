"""Analog quantum sampling solver.

The primary quantum solving primitive that compiles a pulse program and runs it on
a backend to sample bitstrings from the quantum state.
"""

from __future__ import annotations

import dataclasses

import qoolqit
from qoolqit.execution.compilation_functions import CompilerProfile
from qoolqit.execution import job
from qubosolver.types import protocols


def _quantum_program(
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    device: qoolqit.Device,
    *,
    compiler_profile: CompilerProfile = CompilerProfile.MAX_ENERGY,
    default_sequence_duration: int | None = None,
) -> qoolqit.QuantumProgram:
    """Build and compile a :class:`~qoolqit.QuantumProgram` for the target device.

    Constructs a :class:`~qoolqit.QuantumProgram` from *register* and *drive*,
    then calls ``program.compile_to`` with the given *compiler_profile*.

    When the device has no native ``max_duration`` and *default_sequence_duration*
    is given, the device is cloned with that value injected as its
    ``max_sequence_duration`` before compiling, so the sequence duration is
    still bounded. When the device exposes a finite ``max_duration`` constraint
    (whether native or just injected), the sequence duration is capped at 99 %
    of that limit (``device_max_duration_ratio=0.99``) to leave a small safety
    margin and avoid compilation failures at the exact boundary. Otherwise,
    ``device_max_duration_ratio`` is passed as ``None`` and no capping is
    applied.

    Args:
        register: Atom register defining qubit positions.
        drive: Pulse drive schedule encoding the quantum operations.
        device: Target quantum device that provides hardware constraints used
            during compilation.
        compiler_profile: Compilation strategy controlling how the pulse
            sequence is mapped to device constraints.  Defaults to
            ``CompilerProfile.MAX_ENERGY``.
        default_sequence_duration: Fallback maximum sequence duration (ns)
            injected when *device* has no native ``max_duration`` cap.
            ``None`` leaves the device unpatched.

    Returns:
        A compiled :class:`~qoolqit.QuantumProgram` ready to be submitted to
        a backend.
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


def solve(
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    backend: protocols.Backend,
    device: qoolqit.Device,
    *,
    compiler_profile: CompilerProfile = CompilerProfile.MAX_ENERGY,
    default_sequence_duration: int | None = None,
) -> job.Job:
    """Sample bitstrings from an analog quantum program by running it on a backend.

    This is the primary quantum solving primitive — callers obtain measurement
    outcomes by calling ``job.results()`` on the returned job.

    Args:
        register: Atom register defining qubit positions.
        drive: Pulse drive schedule encoding the quantum operations.
        backend: Execution backend.
        device: Target quantum device used for compilation constraints.
        compiler_profile: Compilation strategy.
        default_sequence_duration: Fallback maximum sequence duration (ns).

    Returns:
        A job handle for the submitted execution.  Call [`.results()`][qoolqit.execution.job.Job.results] to retrieve the measurement outcomes once the job completes.
    """
    program = _quantum_program(
        register,
        drive,
        device,
        compiler_profile=compiler_profile,
        default_sequence_duration=default_sequence_duration,
    )

    return backend.run(program)
