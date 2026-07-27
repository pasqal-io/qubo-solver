"""Drive shaping methods for generating quantum drive schedules.

Provides heuristic and optimized strategies for constructing amplitude
and detuning waveforms used to solve QUBO problems on neutral-atom hardware.
"""

from __future__ import annotations


from qubosolver.drive_shaping import heuristic, optimized

__all__ = [
    "heuristic",
    "optimized",
]
