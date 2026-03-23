from __future__ import annotations

from qubosolver import concepts

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

from qoolqit import QuantumProgram
from pulser.backend import Results

# This "test" is intended to be run with mypy, not mypy
# We use # type: ignore[...] in combination with the mypy flag "--warn-unused-ignores"
# to detect whether a class matches the concept when it's expected not to.


def _test_backend(backend: concepts.Backend) -> None: ...


def test_not_backend() -> None:

    class NotBackend0: ...

    _test_backend(NotBackend0())  # type: ignore[arg-type]
    _b0: concepts.Backend = NotBackend0()  # type: ignore[assignment]

    class NotBackend1:
        def run(self: Self, program: QuantumProgram) -> None:
            return

    _test_backend(NotBackend1())  # type: ignore[arg-type]
    _b1: concepts.Backend = NotBackend1()  # type: ignore[assignment]

    class NotBackend2:
        def run(self: Self, program: int) -> Sequence[Results]:
            return [Results(atom_order=(), total_duration=0)]

    _test_backend(NotBackend2())  # type: ignore[arg-type]
    _b2: concepts.Backend = NotBackend2()  # type: ignore[assignment]

    class NotBackend3:
        def run(self: Self, program: QuantumProgram) -> Sequence[Results] | None:
            return [Results(atom_order=(), total_duration=0)]

    _test_backend(NotBackend3())  # type: ignore[arg-type]
    _b3: concepts.Backend = NotBackend3()  # type: ignore[assignment]


def test_backend() -> None:

    class Backend0:
        def run(self: Self, program: QuantumProgram) -> Sequence[Results]:
            return [Results(atom_order=(), total_duration=0)]

    _test_backend(Backend0())
    _b0: concepts.Backend = Backend0()

    class Backend1:
        def run(self: Self, program: QuantumProgram | None) -> Sequence[Results]:
            return [Results(atom_order=(), total_duration=0)]

    _test_backend(Backend1())
    _b1: concepts.Backend = Backend1()
