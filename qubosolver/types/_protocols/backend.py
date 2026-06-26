from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
import qoolqit
from qoolqit.execution import job
from pulser.backend import Results
from .protocol import Protocol


class Backend(Protocol):

    def run(self: Self, program: qoolqit.QuantumProgram) -> job.Job[Results]: ...
