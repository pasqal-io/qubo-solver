"""Drive shaping algorithms for generating quantum drive schedules.

Provides several algorithms for constructing amplitude and detuning waveforms used to solve
QUBO problems on neutral-atom hardware.
"""

from __future__ import annotations

from qubosolver.drive_shaping import (
    proportional_diagonal,
    local_energy_scale,
)

__all__ = [
    "proportional_diagonal",
    "local_energy_scale",
]
