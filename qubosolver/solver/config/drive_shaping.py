from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, get_args

from qubosolver.types import Solution

_DriveShapingAlgorithm = Literal["bayesian_search", "proportional_diagonal", "local_energy_scale"]


@dataclass
class Config():
    """A module-level [`drive_shaping.Config`][] that defines the drive shaping part of a [`solvers.quantum.Config`][].

    Attributes:
        algorithm (_DriveShapingAlgorithm, optional): Drive shaping method used. One of:

            - `"bayesian_search"`: Drive whose parameters are found via Bayesian search that
              minimizes the cost function via pulse optimization.
            - `"proportional_diagonal"`: Drive whose amplitude/detuning scale proportionally to
              the QUBO diagonal; no numerical optimization.
            - `"local_energy_scale"`: Drive whose peak Rabi frequency scales with the average
              local physical energy scale; no numerical optimization.

            Defaults to `"proportional_diagonal"`.
        dmm: Whether to use a detuning map when applying drive shaping or not.
            This adds a [`qoolqit.drive.DetuningMapModulator`][] to the output drive.
            Defaults to `True`, which applies DMM.
        bayesian_search_n_calls: Number of calls for the optimization process.
            Defaults to 20. Note the optimizer accepts a minimal value of 12.
        bayesian_search_initial_omega_parameters: Initial guess for the
            three normalized amplitude waveform knots, each in ``[0, 1]``.
            Defaults to ``[0.5, 0.9, 0.5]``.
        bayesian_search_initial_detuning_parameters: Initial guess for the
            three normalized detuning waveform knots, each in ``[-1, 1]``.
            Defaults to ``[-0.8, 0.0, 0.8]``.
        bayesian_search_seed: Random seed for the Bayesian optimiser.
            Defaults to `None`.
        proportional_diagonal_kappa: Scaling coefficient for the Omega waveform in
            the proportional-diagonal drive shaper. Defaults to `0.25`.
        local_energy_scale_kappa: Scaling coefficient for the Omega waveform in
            the local-energy-scale drive shaper. Defaults to `0.25`.
        default_sequence_duration: Fallback maximum sequence duration
            in nanoseconds injected when the target device has no ``max_duration`` cap.
            Defaults to `50000` ns.
    """

    algorithm: Literal["bayesian_search", "proportional_diagonal", "local_energy_scale"] = "proportional_diagonal"
    dmm: bool = True
    bayesian_search_n_calls: int = 20
    bayesian_search_initial_omega_parameters: list[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.5]
    )
    bayesian_search_initial_detuning_parameters: list[float] = field(
        default_factory=lambda: [-0.8, 0.0, 0.8]
    )
    bayesian_search_seed: int | None = None

    # Proportional-diagonal coefficient for omega
    proportional_diagonal_kappa: float = 0.25

    # Local-energy-scale coefficient for omega
    local_energy_scale_kappa: float = 0.25

    default_sequence_duration: int = 50000

    def __post_init__(self) -> None:
        if self.algorithm not in get_args(_DriveShapingAlgorithm):
            raise ValueError(f"Invalid drive shaping method '{self.algorithm}'.")
        if len(self.bayesian_search_initial_omega_parameters) != 3:
            raise ValueError(
                "`bayesian_search_initial_omega_parameters` should be a list of 3 numbers."
            )
        if len(self.bayesian_search_initial_detuning_parameters) != 3:
            raise ValueError(
                "`bayesian_search_initial_detuning_parameters` should be a list of 3 numbers."
            )
