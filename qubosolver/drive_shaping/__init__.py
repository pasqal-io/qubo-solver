"""Drive shaping algorithms for generating quantum drive schedules.

Provides several algorithms for constructing amplitude and detuning waveforms used to solve
QUBO problems on neutral-atom hardware.
"""

from __future__ import annotations

from qubosolver.drive_shaping import (
    local_energy_scale,
    proportional_diagonal,
)

__all__ = [
    "local_energy_scale",
    "proportional_diagonal",
]
