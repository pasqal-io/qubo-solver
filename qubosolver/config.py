from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import field
from typing import Any, Callable

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, model_serializer
from qoolqit._solvers.data import BackendConfig
from qoolqit._solvers.types import DeviceType
from qoolqit._solvers.types import BackendType

from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    PulseType,
    ClassicalSolverType,
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
    "BackendType",
]


class Config(BaseModel, ABC):
    """Pydantic class for configs."""

    model_config = ConfigDict(extra="forbid")


class ClassicalConfig(Config):
    """A `ClassicalConfig` instance defines the classical
        part of a `SolverConfig`.

    Attributes:
        classical_solver_type (str | ClassicalSolverType, optional): Classical solver type. Defaults to "cplex".
        cplex_maxtime (float, optional): CPLEX maximum runtime. Defaults to 600s.
        cplex_log_path (str, optional): CPLEX log path. Default to `solver.log`.
        max_iter (int, optional): Maximum number of iterations to perform for simulated annealing or tabu search.
        sa_initial_temp (float, optional): Starting temperature (controls exploration).
        sa_final_temp (float, optional): Minimum temperature threshold for stopping.
        sa_alpha (float, optional): Cooling rate - should be slightly below 1 (e.g., 0.95–0.99).
        tabu_x0 (torch.Tensor | None, optional): The initial binary solution tensor of shape (n,).
        tabu_tenure (int, optional): Number of iterations a move (bit flip) remains tabu.
        tabu_max_no_improve (int, optional): Maximum number of consecutive iterations
            without improvement before termination.
    """

    classical_solver_type: str | ClassicalSolverType = "cplex"
    cplex_maxtime: float = 600.0
    cplex_log_path: str = "solver.log"

    max_iter: int = 100

    sa_initial_temp: float = 10.0
    sa_final_temp: float = 0.1
    sa_alpha: float = 0.99

    tabu_x0: torch.Tensor | None = None
    tabu_tenure: int = 7
    tabu_max_no_improve: int = 20

    @field_validator("classical_solver_type")
    @classmethod
    def _normalize_classical_solver_type(
        cls, val: str | ClassicalSolverType
    ) -> ClassicalSolverType | Any:
        """Normalize the classical_solver_type attribute."""
        if isinstance(val, ClassicalSolverType):
            return val
        u = val.upper()
        if u == ClassicalSolverType.CPLEX.name:
            return ClassicalSolverType.CPLEX
        elif u == ClassicalSolverType.TABU_SEARCH.name:
            return ClassicalSolverType.TABU_SEARCH
        elif u == ClassicalSolverType.SIMULATED_ANNEALING.name:
            return ClassicalSolverType.SIMULATED_ANNEALING
        else:
            raise ValueError(f"Invalid classical_solver_type '{val}'.")

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        serialization: dict = {"classical_solver_type": self.classical_solver_type}
        if self.classical_solver_type == "cplex":
            serialization.update(
                {"cplex_maxtime": self.cplex_maxtime, "cplex_log_path": self.cplex_log_path}
            )
        if self.classical_solver_type == "simulated_annealing":
            serialization.update(
                {
                    "max_iter": self.max_iter,
                    "sa_initial_temp": self.sa_initial_temp,
                    "sa_final_temp": self.sa_final_temp,
                    "sa_alpha": self.sa_alpha,
                }
            )
        if self.classical_solver_type == "tabu_search":
            serialization.update(
                {
                    "max_iter": self.max_iter,
                    "tabu_x0": self.tabu_x0,
                    "tabu_tenure": self.tabu_tenure,
                    "tabu_max_no_improve": self.tabu_max_no_improve,
                }
            )
        return serialization


class EmbeddingConfig(Config):
    """A `EmbeddingConfig` instance defines the embedding
        part of a `SolverConfig`.

    Attributes:
        embedding_method (str | EmbedderType | type[BaseEmbedder], optional): The type of
            embedding method used to place atoms on the register according to the QUBO problem.
            Defaults to `EmbedderType.GREEDY`.
        layout_greedy_embedder (LayoutType | str, optional): Layout type for the
            greedy embedder method. Defaults to `LayoutType.TRIANGULAR`.
        blade_steps_per_round (int, optional): The number of steps
            for each layer of dimension for BLaDE.
            Defaults to 200.
        starting_positions (torch.Tensor | None, optional): The starting parameters
            according to the specified dimensions.
            Defaults to None.
        blade_dimensions (list[int], optional): A list of dimension degrees
            to explore one after the other (default is `[5, 4, 3, 2, 2, 2]`).
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
    blade_steps_per_round: int | None = 200
    starting_positions: torch.Tensor | None = None
    blade_dimensions: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
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
            try:
                return EmbedderType[val.upper()]
            except KeyError:
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
