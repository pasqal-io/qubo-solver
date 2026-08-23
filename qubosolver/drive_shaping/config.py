from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

from qubosolver.types import Solution
from .enums import Algorithm


@dataclass
class Config():
    """A `DriveShapingConfig` instance defines the drive shaping part of a `SolverConfig`.

    Attributes:
        algorithm (str | Algorithm, optional): Drive shaping
            method used. Defaults to `Algorithm.PROPORTIONAL_DIAGONAL`.
        dmm (bool, optional): Whether to use a detuning map when applying drive shaping or not.
            This adds WeightedDetuning with a Constant Waveform.
            Defaults to True, which applies DMM.
        bayesian_search_re_execute_opt_drive (bool, optional): Whether to re-run the optimal drive sequence
            after optimization. Defaults to False.
        bayesian_search_n_calls (int, optional): Number of calls for the optimization process.
            Defaults to 20. Note the optimizer accepts a minimal value of 12.
        bayesian_search_initial_omega_parameters (list[float], optional): Initial guess for the
            three normalized amplitude waveform knots, each in ``[0, 1]``.
            Defaults to ``[0.5, 0.9, 0.5]``.
        bayesian_search_initial_detuning_parameters (list[float], optional): Initial guess for the
            three normalized detuning waveform knots, each in ``[-1, 1]``.
            Defaults to ``[-0.8, 0.0, 0.8]``.
        bayesian_search_custom_objective (Callable[[Solution], float], optional):
            For bayesian optimization, one can change the scalar objective
            computed from each evaluation's `Solution` (bitstrings, counts,
            probabilities, and costs) to optimize differently. Instead of using
            the best cost out of the samples, one can change the objective for
            an average, or any function of the form
            `def bayesian_search_custom_objective(solution: Solution) -> float`.
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        bayesian_search_seed (int | None): Random seed for the Bayesian optimiser.
            Defaults to None.
        proportional_diagonal_kappa (float): Scaling coefficient for the Omega waveform in
            the proportional-diagonal drive shaper. Defaults to 0.25.
        local_energy_scale_kappa (float): Scaling coefficient for the Omega waveform in
            the local-energy-scale drive shaper. Defaults to 0.25.
        default_sequence_duration (int, optional): Fallback maximum sequence duration
            (ns) injected when the target device has no ``max_duration`` cap.
            Defaults to 50000.
    """

    algorithm: Algorithm | str = Algorithm.PROPORTIONAL_DIAGONAL
    dmm: bool = True
    bayesian_search_n_calls: int = 20
    bayesian_search_initial_omega_parameters: list[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.5]
    )
    bayesian_search_initial_detuning_parameters: list[float] = field(
        default_factory=lambda: [
            -0.8,
            0.0,
            0.8,
        ]
    )  # ---> default initial drive parameters: delta = (-2, 0, 2)
    bayesian_search_custom_objective: Callable[[Solution], float] | None = None
    bayesian_search_seed: int | None = None
    bayesian_search_re_execute_opt_drive: bool = False

    # Proportional-diagonal coefficient for omega
    proportional_diagonal_kappa: float = 0.25

    # Local-energy-scale coefficient for omega
    local_energy_scale_kappa: float = 0.25

    default_sequence_duration: int = 50000

    def __post_init__(self) -> None:
        self.algorithm = self._normalize_drive_shaping_method(self.algorithm)
        if len(self.bayesian_search_initial_omega_parameters) != 3:
            raise ValueError(
                "`bayesian_search_initial_omega_parameters` should be a list of 3 numbers."
            )
        if len(self.bayesian_search_initial_detuning_parameters) != 3:
            raise ValueError(
                "`bayesian_search_initial_detuning_parameters` should be a list of 3 numbers."
            )

    @staticmethod
    def _normalize_drive_shaping_method(val: str | Algorithm) -> Algorithm:
        """Normalize the `drive_shaping_method` attribute."""
        if isinstance(val, Algorithm):
            return val
        elif isinstance(val, str):
            try:
                return Algorithm[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid drive shaping method '{val}'.")
        else:
            raise TypeError("Invalid drive shaping method type.")
