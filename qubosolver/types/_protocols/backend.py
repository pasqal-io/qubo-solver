from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
import qoolqit
from qoolqit.execution import job
from pulser.backend import Results
from .protocol import Protocol


class Backend(Protocol):
    """Structural protocol for quantum backends.

    Any object that implements :meth:`run` with the expected signature
    is considered a ``Backend``, without requiring explicit subclassing.
    This enables duck-typed compatibility checks (via ``isinstance`` when
    beartype runtime checking is active, or static type checking via mypy/pyright).

    Typical implementations wrap a remote or local quantum execution engine
    (e.g. a Qoolqit emulator or a real QPU) and return a handle to the
    asynchronous computation.
    """

    def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results]:
        """Submit a quantum program for execution.

        Args:
            program: The compiled quantum program to execute.

        Returns:
            A :class:`~qoolqit.execution.job.Job` wrapping the
            :class:`~pulser.backend.Results` that will be populated
            upon completion.
        """
        ...
