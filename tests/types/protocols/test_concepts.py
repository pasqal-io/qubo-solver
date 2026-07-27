from __future__ import annotations

from qubosolver.types import _protocols

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

import qoolqit
from qoolqit.execution import job
from pulser.backend import Results

# This "test" is intended to be run with mypy, not mypy
# We use # type: ignore[...] in combination with the mypy flag "--warn-unused-ignores"
# to detect whether a class matches the concept when it's expected not to.


def _test_backend(backend: _protocols.Backend) -> None: ...


def test_not_backend() -> None:

    class NotBackend0: ...

    _test_backend(NotBackend0())  # type: ignore[arg-type]
    _b0: _protocols.Backend = NotBackend0()  # type: ignore[assignment]

    class NotBackend1:
        def run(self: Self, program: qoolqit.QuantumProgram) -> None:
            return

    _test_backend(NotBackend1())  # type: ignore[arg-type]
    _b1: _protocols.Backend = NotBackend1()  # type: ignore[assignment]

    class NotBackend2:
        def run(self: Self, program: int) -> job.Job[Results]:
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(NotBackend2())  # type: ignore[arg-type]
    _b2: _protocols.Backend = NotBackend2()  # type: ignore[assignment]

    class NotBackend3:
        def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results] | None:
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(NotBackend3())  # type: ignore[arg-type]
    _b3: _protocols.Backend = NotBackend3()  # type: ignore[assignment]


def test_backend() -> None:

    class Backend0:
        def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results]:
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(Backend0())
    _b0: _protocols.Backend = Backend0()

    class Backend1:
        def run(self: Self, program: qoolqit.QuantumProgram | None) -> job.Job[Results]:
            return job._LocalJob(Results(atom_order=(), total_duration=0))

    _test_backend(Backend1())
    _b1: _protocols.Backend = Backend1()
