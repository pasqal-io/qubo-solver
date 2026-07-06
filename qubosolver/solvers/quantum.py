"""Analog quantum sampling solver.

The public entry point of this module is `analog_quantum_sample` — the
primary quantum solving primitive that compiles a pulse program and runs it on
a backend to sample bitstrings from the quantum state.  It is exported via
:mod:`qubosolver.solvers` and called by
:class:`~qubosolver.solvers.BaseSolver`.
"""

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
    """Build and compile a :class:`~qoolqit.QuantumProgram` for the target device.

    Constructs a :class:`~qoolqit.QuantumProgram` from *register* and *drive*,
    then calls ``program.compile_to`` with the given *compiler_profile*.

    When the device exposes a finite ``max_duration`` constraint, the sequence
    duration is capped at 99 % of that limit (``device_max_duration_ratio=0.99``)
    to leave a small safety margin and avoid compilation failures at the exact
    boundary.  When no duration limit is set, ``device_max_duration_ratio`` is
    passed as ``None`` and no capping is applied.

    Args:
        register: Atom register defining qubit positions.
        drive: Pulse drive schedule encoding the quantum operations.
        device: Target quantum device that provides hardware constraints used
            during compilation.
        compiler_profile: Compilation strategy controlling how the pulse
            sequence is mapped to device constraints.  Use
            ``CompilerProfile.MAX_ENERGY`` (default) for heuristic
            drive-shaping, and ``CompilerProfile.WORKING_POINT`` for the
            Bayesian-optimised drive-shaping path.

    Returns:
        A compiled :class:`~qoolqit.QuantumProgram` ready to be submitted to
        a backend.
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
    """Sample bitstrings from an analog quantum program by running it on a backend.

    This is the primary quantum solving primitive — callers obtain measurement
    outcomes by calling ``job.results()`` on the returned job.

    Args:
        register: Atom register defining qubit positions.
        drive: Pulse drive schedule encoding the quantum operations.
        backend: Execution backend.
        device: Target quantum device used for compilation constraints.
        compiler_profile: Compilation strategy forwarded to
            `_quantum_program`.  Defaults to ``MAX_ENERGY``; use
            ``WORKING_POINT`` for the optimised drive-shaping path.

    Returns:
        A :class:`~qoolqit.execution.job.Job` handle for the submitted
        execution.  Call ``job.results()`` to retrieve the measurement
        outcomes once the job completes.
    """
    program = _quantum_program(register, drive, device, compiler_profile=compiler_profile)

    return backend.run(program)
