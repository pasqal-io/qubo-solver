from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import field
from typing import Any, Callable

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from qoolqit._solvers.data import BackendConfig
from qoolqit._solvers.types import DeviceType

from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    PulseType,
)

# to handle torch Tensor
BaseModel.model_config["arbitrary_types_allowed"] = True

# Modules to be automatically added to the qubosolver namespace
__all__: list[str] = [
    "ClassicalConfig",
    "EmbeddingConfig",
    "PulseShapingConfig",
    "BackendConfig",
    "SolverConfig",
]


class Config(BaseModel, ABC):
    """Pydantic class for configs."""

    model_config = ConfigDict(extra="forbid")


class ClassicalConfig(Config):
    """A `ClassicalConfig` instance defines the classical
        part of a `SolverConfig`.

    Attributes:
        classical_solver_type (str, optional): Classical solver type. Defaults to "cplex".
        cplex_maxtime (float, optional): CPLEX maximum runtime. Defaults to 600s.
        cplex_log_path (str, optional): CPLEX log path. Default to `solver.log`.
    """

    classical_solver_type: str = "cplex"
    cplex_maxtime: float = 600.0
    cplex_log_path: str = "solver.log"


class EmbeddingConfig(Config):
    """A `EmbeddingConfig` instance defines the embedding
        part of a `SolverConfig`.

    Attributes:
        embedding_method (str | EmbedderType | type[BaseEmbedder], optional): The type of
            embedding method used to place atoms on the register according to the QUBO problem.
            Defaults to `EmbedderType.GREEDY`.
        layout_greedy_embedder (LayoutType | str, optional): Layout type for the
            greedy embedder method. Defaults to `LayoutType.TRIANGULAR`.
        traps (int, optional): The number of traps on the register.
            Defaults to `DeviceType.ANALOG_DEVICE.value.min_layout_traps`.
        spacing (float, optional): The minimum distance between atoms.
            Defaults to `DeviceType.ANALOG_DEVICE.value.min_atom_distance`.
        density (float, optional): The estimated density of the QUBO matrix.
            Defaults to None.
        draw_steps (bool, optional): Show generated graph at each step of the optimization.
            Defaults to `False`.
        animation_save_path (str | None, optional): If provided, path to save animation.
            Defaults to None.
    """

    embedding_method: Any = EmbedderType.GREEDY
    layout_greedy_embedder: LayoutType | str = LayoutType.TRIANGULAR
    traps: int = DeviceType.DIGITAL_ANALOG_DEVICE.value.min_layout_traps
    spacing: float = float(DeviceType.DIGITAL_ANALOG_DEVICE.value.min_atom_distance)
    density: float | None = None
    draw_steps: bool = False
    animation_save_path: str | None = None

    @field_validator("embedding_method")
    @classmethod
    def _normalize_embedder(cls, val: Any) -> EmbedderType | Any:
        """Normalize the embedded attribute."""
        if isinstance(val, EmbedderType):
            return val
        elif isinstance(val, str):
            u = val.upper()
            if u == EmbedderType.GREEDY.name:
                return EmbedderType.GREEDY
            else:
                raise ValueError(f"Invalid str embedding method '{val}'.")
        elif inspect.isclass(val):
            from qubosolver.pipeline.embedder import BaseEmbedder

            if not issubclass(val, BaseEmbedder):
                raise TypeError("Class must be a subclass of BaseEmbedder")
            else:
                return val
        else:
            raise TypeError("Invalid embedding method type.")

    @field_validator("layout_greedy_embedder")
    @classmethod
    def _normalize_layout(cls, val: str | LayoutType) -> LayoutType:
        """Normalize the layout attribute."""
        if isinstance(val, LayoutType):
            return val
        u = val.upper()
        if u == LayoutType.SQUARE.name:
            return LayoutType.SQUARE
        elif u == LayoutType.TRIANGULAR.name:
            return LayoutType.TRIANGULAR
        else:
            raise ValueError(f"Invalid layout '{val}'.")


class PulseShapingConfig(Config):
    """A `PulseShapingConfig` instance defines the pulse shaping
        part of a `SolverConfig`.

    Attributes:
        pulse_shaping_method (str | PulseType | type[BasePulseShaper], optional): Pulse shaping
            method used. Defauts to `PulseType.ADIABATIC`.
        initial_omega_parameters (List[float], optional): Default initial omega parameters
            for the pulse. Defaults to Omega = (5, 10, 5).
        initial_detuning_parameters (List[float], optional): Default initial detuning parameters
            for the pulse. Defaults to delta = (-10, 0, 10).
        re_execute_opt_pulse (bool, optional): Whether to re-run the optimal pulse sequence.
            Defaults to False.
        custom_qubo_cost (Callable[[str, torch.Tensor], float], optional): Apply a different
            qubo cost evaluation
            than the default QUBO evaluation defined in
            `qubosolver/pipeline/pulse.py:OptimizedPulseShaper.compute_qubo_cost`.
            Must be defined as:
            `def custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`.
            Defaults to None, meaning we use the default QUBO evaluation.
        custom_objective_fn (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `qubosolver/pipeline/pulse.py:OptimizedPulseShaper.run_simulation`
            to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = custom_objective_fn(bitstrings,
                counts, probabilities, costs, best_cost, best_bitstring)`
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        callback_objective (Callable[..., None], optional): Apply a callback
            during bayesian optimization. Only accepts one input dictionary
            created during optimization `d = {"x": x, "cost_eval": cost_eval}`
            hence should be defined as:
            `def callback_fn(d: dict) -> None:`
            Defaults to None, which means no callback is applied.
    """

    pulse_shaping_method: Any = PulseType.ADIABATIC
    initial_omega_parameters: list[float] = field(
        default_factory=lambda: [
            5.0,
            10.0,
            5.0,
        ]
    )  # ---> default initial pulse parameters: Omega = (5, 10, 5)
    initial_detuning_parameters: list[float] = field(
        default_factory=lambda: [
            -10.0,
            0.0,
            10.0,
        ]
    )  # ---> default initial pulse parameters: delta = (-10, 0, 10)
    re_execute_opt_pulse: bool = False
    custom_qubo_cost: Callable[[str, torch.Tensor], float] | None = None
    custom_objective: Callable[[list, list, list, list, float, str], float] | None = None
    callback_objective: Callable[..., None] | None = None

    @field_validator("pulse_shaping_method")
    @classmethod
    def _normalize_pulse_shaping_method(cls, val: Any) -> PulseType | Any:
        """Normalize the `pulse_shaping_method` attribute."""
        if isinstance(val, PulseType):
            return val
        elif isinstance(val, str):
            u = val.upper()
            if u == PulseType.ADIABATIC.name:
                return PulseType.ADIABATIC
            elif u == PulseType.OPTIMIZED.name:
                return PulseType.OPTIMIZED
            else:
                raise ValueError(f"Invalid pulse shaping method '{val}'.")

        elif inspect.isclass(val):
            from qubosolver.pipeline.pulse import BasePulseShaper

            if not issubclass(val, BasePulseShaper):
                raise TypeError("Class must be a subclass of BasePulseShaper")
            else:
                return val
        else:
            raise TypeError("Invalid pulse shaping method type.")

    @field_validator("initial_omega_parameters")
    @classmethod
    def _check_initial_omega_parameters(cls, val: list[float]) -> list[float]:
        if len(val) == 3:
            return val
        else:
            raise ValueError("`initial_omega_parameters` should be a list of 3 numbers.")

    @field_validator("initial_detuning_parameters")
    @classmethod
    def _check_initial_detuning_parameters(cls, val: list[float]) -> list[float]:
        if len(val) == 3:
            return val
        else:
            raise ValueError("`initial_detuning_parameters` should be a list of 3 numbers.")


class SolverConfig(Config):
    """
    A `SolverConfig` instance defines how a QUBO problem should be solved.
    We specify whether to use a quantum or classical approach,
    which backend to run on, and additional execution parameters.

    Attributes:
        config_name (str, optional): The name of the current configuration.
            Defaults to ''.
        use_quantum (bool, optional): Whether to solve using a quantum approach (`True`)
            or a classical approach (`False`). Defaults to False.
        backend (BackendConfig, optional): Which underlying backend configuration is used.
            Defaults to the default BackendConfig using `BackendType.QUTIP`.

        n_calls (int, optional): Number of calls for the optimization process inside VQA.
            Defaults to 20. Note the optimizer accepts a minimal value of 12.
        embedding (EmbeddingConfig, optional): Embedding part configuration of the solver.
        pulse_shaping (PulseShapingConfig, optional): Pulse-shaping part configuration
            of the solver.
        classical (ClassicalConfig, optional): Classical part configuration of the solver.

        num_shots (int, optional): Number of samples. Defaults to 500.

        do_postprocessing (bool, optional): Whether we apply post-processing (`True`)
            or not (`False`).
        do_preprocessing (bool, optional): Whether we apply pre-processing (`True`)
            or not (`False`)
    """

    config_name: str = ""
    use_quantum: bool | None = False
    backend_config: BackendConfig = BackendConfig()
    n_calls: int = 20
    embedding: EmbeddingConfig = EmbeddingConfig()
    pulse_shaping: PulseShapingConfig = PulseShapingConfig()
    classical: ClassicalConfig = ClassicalConfig()
    num_shots: int = 500
    do_postprocessing: bool = False
    do_preprocessing: bool = False
    activate_trivial_solutions: bool = True

    def __repr__(self) -> str:
        return self.config_name

    def _specs(self) -> str:
        """Return the specs of the `SolverConfig`, that is all attributes.

        Returns:
            dict: Dictionary of specs key-values.
        """
        return "\n".join(
            f"{k}: ''" if v == "" else f"{k}: {v}" for k, v in self.model_dump().items()
        )

    def print_specs(self) -> None:
        """Print specs."""
        print(self._specs())

    @model_validator(mode="after")
    def _set_traps_spacing_from_device(self) -> SolverConfig:

        if self.backend_config.device:
            device = self.backend_config.device
            if hasattr(device, "value"):
                if device.value.min_layout_traps:
                    if self.embedding.traps < device.value.min_layout_traps:
                        self.embedding = self.embedding.model_copy(
                            update={"traps": device.value.min_layout_traps}
                        )
                if device.value.min_atom_distance:
                    spacing_device = float(device.value.min_atom_distance)
                    if self.embedding.spacing < spacing_device:
                        self.embedding = self.embedding.model_copy(
                            update={"spacing": spacing_device}
                        )
        return self

    @classmethod
    def from_kwargs(cls, **kwargs: dict) -> SolverConfig:
        """Create an instance based on entries of other configs.

        Note that if any of the keywords
        ("backend_config", "embedding", "pulse_shaping", "classical")
        are present in kwargs, the values are taken directly.

        Returns:
            SolverConfig: An instance from values.
        """
        # Extract fields from pydantic BaseModel
        backend_config_fields = {k: v for k, v in kwargs.items() if k in BackendConfig.model_fields}
        embedding_fields = {k: v for k, v in kwargs.items() if k in EmbeddingConfig.model_fields}
        pulse_shaping_fields = {
            k: v for k, v in kwargs.items() if k in PulseShapingConfig.model_fields
        }
        classical_fields = {k: v for k, v in kwargs.items() if k in ClassicalConfig.model_fields}

        solver_fields = {
            k: v
            for k, v in kwargs.items()
            if k in cls.model_fields
            and k not in ("backend_config", "embedding", "pulse_shaping", "classical")
        }

        return cls(
            backend_config=(
                BackendConfig(**backend_config_fields)
                if "backend_config" not in kwargs
                else kwargs["backend_config"]
            ),
            embedding=(
                EmbeddingConfig(**embedding_fields)
                if "embedding" not in kwargs
                else kwargs["embedding"]
            ),
            pulse_shaping=(
                PulseShapingConfig(**pulse_shaping_fields)
                if "pulse_shaping" not in kwargs
                else kwargs["pulse_shaping"]
            ),
            classical=(
                ClassicalConfig(**classical_fields)
                if "classical" not in kwargs
                else kwargs["classical"]
            ),
            **solver_fields,
        )
