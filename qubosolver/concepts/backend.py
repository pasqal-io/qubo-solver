from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

from qoolqit import QuantumProgram
from qoolqit.execution import job
from pulser.backend import Results


class Backend(Protocol):

    def run(self: Self, program: QuantumProgram) -> job.Job[Results]: ...
