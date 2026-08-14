from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

from qubosolver.types import Bitstring, Matrix, Solution
from qubosolver.utils._config import _Config
from . import Algorithm


@dataclass
class Config(_Config):
    """A `DriveShapingConfig` instance defines the drive shaping part of a `SolverConfig`.

    Attributes:
        drive_shaping_method (str | Algorithm | type[BaseDriveShaper], optional): Drive shaping
            method used. Defaults to `Algorithm.PROPORTIONAL_DIAGONAL`.
        dmm (bool, optional): Whether to use a detuning map when applying drive shaping or not.
            This adds WeightedDetuning with a Constant Waveform.
            Defaults to True, which applies DMM.
        bayesian_search_re_execute_opt_drive (bool, optional): Whether to re-run the optimal drive sequence
            after optimization. Defaults to False.
        bayesian_search_n_calls (int, optional): Number of calls for the optimization process.
            Defaults to 20. Note the optimizer accepts a minimal value of 12.
        bayesian_search_initial_omega_parameters (list[float], optional): Default initial omega parameters
            for the drive. Defaults to Omega = (1, 2, 1).
        bayesian_search_initial_detuning_parameters (list[float], optional): Default initial detuning parameters
            for the drive. Defaults to delta = (-2, 0, 2).
        bayesian_search_custom_qubo_cost (Callable[[str, torch.Tensor], float], optional): Apply a different
            qubo cost evaluation
            than the default QUBO evaluation defined in
            `qubosolver/pipeline/drive.py:BayesianSearchDriveShaper.compute_qubo_cost`.
            Must be defined as:
            `def bayesian_search_custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`.
            Defaults to None, meaning we use the default QUBO evaluation.
        bayesian_search_custom_objective (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `qubosolver/pipeline/drive.py:BayesianSearchDriveShaper.run_simulation`
            to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = bayesian_search_custom_objective(bitstrings,
                counts, probabilities, costs, best_cost, best_bitstring)`
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        bayesian_search_callback_objective (Callable[..., None], optional): Apply a callback
            during bayesian optimization. Only accepts one input dictionary
            created during optimization `d = {"x": x, "cost_eval": cost_eval}`
            hence should be defined as:
            `def callback_fn(d: dict) -> None:`
            Defaults to None, which means no callback is applied.
        bayesian_search_seed (int | None): Random seed for the Bayesian optimiser.
            Defaults to None.
        proportional_diagonal_kappa (float): Scaling coefficient for the Omega waveform in
            the proportional-diagonal drive shaper. Defaults to 0.25.
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
    bayesian_search_custom_qubo_cost: Callable[[Bitstring, Matrix], float] | None = None
    bayesian_search_custom_objective: Callable[[Solution], float] | None = None
    bayesian_search_callback_objective: Callable[..., None] | None = None
    bayesian_search_seed: int | None = None
    bayesian_search_re_execute_opt_drive: bool = False

    # Proportional-diagonal coefficient for omega
    proportional_diagonal_kappa: float = 0.25

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the fields relevant to the active drive shaping method.

        Always includes ``drive_shaping_method`` and ``dmm``. When
        ``drive_shaping_method`` is ``BAYESIAN_SEARCH``, all ``bayesian_search_*`` fields
        are also included. Proportional-diagonal-only fields are omitted for the
        Bayesian-search path and vice-versa.

        Returns:
            dict[str, Any]: Serialized representation of this config.
        """
        serialization: dict = {
            "drive_shaping_method": self.algorithm,
            "dmm": self.dmm,
        }
        if self.algorithm == Algorithm.BAYESIAN_SEARCH:
            dict_all_fields = self.__dict__
            serialization.update(
                {
                    k: v
                    for k, v in dict_all_fields.items()
                    if k.startswith(Algorithm.BAYESIAN_SEARCH.value)
                }
            )
        return serialization

    @staticmethod
    def _normalize_drive_shaping_method(val: Any) -> Algorithm | Any:
        """Normalize the `drive_shaping_method` attribute."""
        if isinstance(val, Algorithm):
            return val
        elif isinstance(val, str):
            u = val.upper()
            if u == Algorithm.PROPORTIONAL_DIAGONAL.name:
                return Algorithm.PROPORTIONAL_DIAGONAL
            elif u == Algorithm.BAYESIAN_SEARCH.name:
                return Algorithm.BAYESIAN_SEARCH
            else:
                raise ValueError(f"Invalid drive shaping method '{val}'.")
        elif inspect.isclass(val):
            from qubosolver.drive_shaping._drive_shaper import _BaseDriveShaper

            if not issubclass(val, _BaseDriveShaper):
                raise TypeError(f"Class must be a subclass of {_BaseDriveShaper.__name__}")
            else:
                return val
        else:
            raise TypeError("Invalid drive shaping method type.")
