"""Drive shaping methods for generating quantum drive schedules.

Provides proportional-diagonal, local-energy-scale, and Bayesian-search
strategies for constructing amplitude and detuning waveforms used to solve
QUBO problems on neutral-atom hardware.
"""

from __future__ import annotations

from qubosolver.drive_shaping import (
    proportional_diagonal,
    _local_energy_scale_drive,
    bayesian_search,
)

__all__ = [
    "proportional_diagonal",
    "_local_energy_scale_drive",
    "bayesian_search",
]