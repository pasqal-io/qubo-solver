"""Enum types used by the drive shaping module."""

from __future__ import annotations

from qubosolver.types.enums import _StrEnum


class Algorithm(_StrEnum):
    """Type of drive shaping method applied to the analog quantum pulse sequence."""

    BAYESIAN_SEARCH = "bayesian_search"
    """Drive whose parameters are found via Bayesian search that minimizes the cost function via pulse optimization."""
    PROPORTIONAL_DIAGONAL = "proportional_diagonal"
    """Drive whose amplitude/detuning scale proportionally to the QUBO diagonal; no numerical optimization."""
