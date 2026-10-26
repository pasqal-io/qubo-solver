# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

    import qoolqit

    from qubosolver.types import protocols

from pulser.backend import Results
from qoolqit.execution import job

# This "test" is intended to be run with mypy, not mypy
# We use # type: ignore[...] in combination with the mypy flag "--warn-unused-ignores"
# to detect whether a class matches the concept when it's expected not to.


def _test_backend(backend: protocols.Backend) -> None: ...


def test_not_backend() -> None:

    class NotBackend0: ...

    _test_backend(NotBackend0())  # type: ignore[arg-type]
    _b0: protocols.Backend = NotBackend0()  # type: ignore[assignment]

    class NotBackend1:
        def run(self: Self, program: qoolqit.QuantumProgram) -> None:  # noqa: ARG002
            return

    _test_backend(NotBackend1())  # type: ignore[arg-type]
    _b1: protocols.Backend = NotBackend1()  # type: ignore[assignment]

    class NotBackend2:
        def run(self: Self, program: int) -> job.Job[Results]:  # noqa: ARG002
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(NotBackend2())  # type: ignore[arg-type]
    _b2: protocols.Backend = NotBackend2()  # type: ignore[assignment]

    class NotBackend3:
        def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results] | None:  # noqa: ARG002
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(NotBackend3())  # type: ignore[arg-type]
    _b3: protocols.Backend = NotBackend3()  # type: ignore[assignment]


def test_backend() -> None:

    class Backend0:
        def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results]:  # noqa: ARG002
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(Backend0())
    _b0: protocols.Backend = Backend0()

    class Backend1:
        def run(self: Self, program: qoolqit.QuantumProgram | None) -> job.Job[Results]:  # noqa: ARG002
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(Backend1())
    _b1: protocols.Backend = Backend1()
