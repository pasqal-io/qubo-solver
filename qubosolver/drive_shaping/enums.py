"""Enum types used by the drive shaping module."""

from __future__ import annotations

from qubosolver.types._enums import _StrEnum


class Algorithm(_StrEnum):
    """Type of drive shaping method applied to the analog quantum pulse sequence."""

    BAYESIAN_SEARCH = "bayesian_search"
    """Drive whose parameters are found via Bayesian search that minimizes the cost function via pulse optimization."""
    PROPORTIONAL_DIAGONAL = "proportional_diagonal"
    """Drive whose amplitude/detuning scale proportionally to the QUBO diagonal; no numerical optimization."""
    LOCAL_ENERGY_SCALE = "local_energy_scale"
    """Drive whose peak Rabi frequency scales with the average local physical energy scale; no numerical optimization."""
