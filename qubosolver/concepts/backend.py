from __future__ import annotations

from typing import Protocol, Sequence, Self
from qoolqit import QuantumProgram
from pulser.backend import Results


class Backend(Protocol):

    def run(self: Self, program: QuantumProgram) -> Sequence[Results]: ...
