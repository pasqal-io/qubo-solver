from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import field
from typing import Any, Callable

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, model_serializer

from pulser_simulation import QutipBackendV2
from pulser.devices import DigitalAnalogDevice as PulserDigitalAnalogDevice
from qoolqit.devices.device import Device, DigitalAnalogDevice
from qoolqit.execution import LocalEmulator, RemoteEmulator, QPU
from pulser_pasqal import PasqalCloud

from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    DriveType,
    ClassicalSolverType,
)

# to handle torch Tensor
BaseModel.model_config["arbitrary_types_allowed"] = True

# Modules to be automatically added to the qubosolver namespace
__all__: list[str] = [
    "LocalEmulator",
    "RemoteEmulator",
    "QPU",
    "PasqalCloud",
    "ClassicalConfig",
    "EmbeddingConfig",
    "DriveShapingConfig",
    "SolverConfig",
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
        max_bitstrings (int, optional): Maximal number of bitstrings returned as solutions.
        sa_initial_temp (float, optional): Starting temperature (controls exploration).
        sa_final_temp (float, optional): Minimum temperature threshold for stopping.
        sa_cooling_rate (float, optional): Cooling rate - should be slightly below 1 (e.g., 0.95–0.99).
        sa_seed (int, optional): Random seed for reproducibility.
        sa_start (torch.Tensor | None, optional): Optioanl initial bitstring of shape (n,).
        sa_energy_tol (float, optional): Energy tolerance for considering two solutions as equivalent.
        tabu_x0 (torch.Tensor | None, optional): The initial binary solution tensor of shape (n,).
        tabu_tenure (int, optional): Number of iterations a move (bit flip) remains tabu.
        tabu_max_no_improve (int, optional): Maximum number of consecutive iterations
            without improvement before termination.
    """

    classical_solver_type: str | ClassicalSolverType = "cplex"
    cplex_maxtime: float = 600.0
    cplex_log_path: str = "solver.log"

    max_iter: int = 100
    max_bitstrings: int = 1

    sa_initial_temp: float = 10.0
    sa_final_temp: float = 0.1
    sa_cooling_rate: float | None = None
    sa_seed: int | None = None
    sa_start: torch.Tensor | None = None
    sa_energy_tol: float = 0.0

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
        all_names = [c.name for c in ClassicalSolverType]
        if u in all_names:
            return ClassicalSolverType[u]
        else:
            raise ValueError(f"Invalid classical_solver_type '{val}'.")

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        serialization: dict = {"classical_solver_type": self.classical_solver_type}
        if self.classical_solver_type == ClassicalSolverType.CPLEX:
            serialization.update(
                {"cplex_maxtime": self.cplex_maxtime, "cplex_log_path": self.cplex_log_path}
            )
        if self.classical_solver_type == ClassicalSolverType.SIMULATED_ANNEALING:
            serialization.update(
                {
                    "max_iter": self.max_iter,
                    "max_bitstrings": self.max_bitstrings,
                    "sa_initial_temp": self.sa_initial_temp,
                    "sa_final_temp": self.sa_final_temp,
                    "sa_cooling_rate": self.sa_cooling_rate,
                    "sa_seed": self.sa_seed,
                    "sa_start": self.sa_start,
                    "sa_energy_tol": self.sa_energy_tol,
                }
            )
        if self.classical_solver_type == ClassicalSolverType.TABU_SEARCH:
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
        greedy_layout (LayoutType | str, optional): Layout type for the
            greedy embedder method. Defaults to `LayoutType.TRIANGULAR`.
        greedy_traps (int, optional): The number of traps on the register.
            Defaults to `pulser.DigitalAnalogDevice.min_layout_traps`.
        greedy_spacing (float, optional): The minimum distance between atoms.
            Defaults to `pulser.PulserDigitalAnalogDevice.min_atom_distance`.
        greedy_density (float, optional): The estimated density of the QUBO matrix.
            Defaults to None.
        blade_steps_per_round (int, optional): The number of steps
            for each layer of dimension for BLaDE.
            Defaults to 200.
        blade_starting_positions (torch.Tensor | None, optional): The starting parameters
            according to the specified dimensions.
            Defaults to None.
        blade_dimensions (list[int], optional): A list of dimension degrees
            to explore one after the other (default is `[5, 4, 3, 2, 2, 2]`).
        draw_steps (bool, optional): Show generated graph at each step of the optimization.
            Defaults to `False`.
        animation_save_path (str | None, optional): If provided, path to save animation.
            Defaults to None.
    """

    embedding_method: Any = EmbedderType.GREEDY
    greedy_layout: LayoutType | str = LayoutType.TRIANGULAR
    greedy_traps: int = PulserDigitalAnalogDevice.min_layout_traps
    greedy_spacing: float = float(PulserDigitalAnalogDevice.min_atom_distance)
    greedy_density: float | None = None
    blade_steps_per_round: int | None = 200
    blade_starting_positions: torch.Tensor | None = None
    blade_dimensions: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    draw_steps: bool = False
    animation_save_path: str | None = None

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        serialization: dict = {
            "embedding_method": self.embedding_method,
            "draw_steps": self.draw_steps,
            "animation_save_path": self.animation_save_path,
        }

        dict_all_fields = self.__dict__
        if self.embedding_method == EmbedderType.GREEDY:
            serialization.update(
                {
                    k: v
                    for k, v in dict_all_fields.items()
                    if k.startswith(EmbedderType.GREEDY.value)
                }
            )

        if self.embedding_method == EmbedderType.BLADE:
            serialization.update(
                {k: v for k, v in dict_all_fields.items() if k.startswith(EmbedderType.BLADE.value)}
            )
        return serialization

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
                raise TypeError(f"Class must be a subclass of {BaseEmbedder.__name__}")
            else:
                return val
        else:
            raise TypeError("Invalid embedding method type.")

    @field_validator("greedy_layout")
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


class DriveShapingConfig(Config):
    """A `DriveShapingConfig` instance defines the drive shaping part of a `SolverConfig`.

    Attributes:
        drive_shaping_method (str | DriveType | type[BaseDriveShaper], optional): Drive shaping
            method used. Defauts to `DriveType.ADIABATIC`.
        dmm (bool, optional): Whether to use a detuning map when applying drive shaping or not.
            This adds WeightedDetuning with a Constant Waveform.
            Defaults to True, which applies DMM.
        optimized_re_execute_opt_drive (bool, optional): Whether to re-run the optimal drive sequence
            after optimization. Defaults to False.
        optimized_n_calls (int, optional): Number of calls for the optimization process.
            Defaults to 20. Note the optimizer accepts a minimal value of 12.
        optimized_initial_omega_parameters (List[float], optional): Default initial omega parameters
            for the drive. Defaults to Omega = (5, 10, 5).
        optimized_initial_detuning_parameters (List[float], optional): Default initial detuning parameters
            for the drive. Defaults to delta = (-10, 0, 10).
        optimized_custom_qubo_cost (Callable[[str, torch.Tensor], float], optional): Apply a different
            qubo cost evaluation
            than the default QUBO evaluation defined in
            `qubosolver/pipeline/drive.py:OptimizedDriveShaper.compute_qubo_cost`.
            Must be defined as:
            `def optimized_custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`.
            Defaults to None, meaning we use the default QUBO evaluation.
        optimized_custom_objective_fn (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `qubosolver/pipeline/drive.py:OptimizedDriveShaper.run_simulation`
            to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = optimized_custom_objective_fn(bitstrings,
                counts, probabilities, costs, best_cost, best_bitstring)`
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        optimized_callback_objective (Callable[..., None], optional): Apply a callback
            during bayesian optimization. Only accepts one input dictionary
            created during optimization `d = {"x": x, "cost_eval": cost_eval}`
            hence should be defined as:
            `def callback_fn(d: dict) -> None:`
            Defaults to None, which means no callback is applied.
    """

    drive_shaping_method: Any = DriveType.ADIABATIC
    dmm: bool = True
    optimized_re_execute_opt_drive: bool = False
    optimized_n_calls: int = 20
    optimized_initial_omega_parameters: list[float] = field(
        default_factory=lambda: [
            5.0,
            10.0,
            5.0,
        ]
    )  # ---> default initial drive parameters: Omega = (5, 10, 5)
    optimized_initial_detuning_parameters: list[float] = field(
        default_factory=lambda: [
            -10.0,
            0.0,
            10.0,
        ]
    )  # ---> default initial drive parameters: delta = (-10, 0, 10)
    optimized_custom_qubo_cost: Callable[[str, torch.Tensor], float] | None = None
    optimized_custom_objective: Callable[[list, list, list, list, float, str], float] | None = None
    optimized_callback_objective: Callable[..., None] | None = None

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        serialization: dict = {
            "drive_shaping_method": self.drive_shaping_method,
            "dmm": self.dmm,
        }
        if self.drive_shaping_method == DriveType.OPTIMIZED:
            dict_all_fields = self.__dict__
            serialization.update(
                {
                    k: v
                    for k, v in dict_all_fields.items()
                    if k.startswith(DriveType.OPTIMIZED.value)
                }
            )
        return serialization

    @field_validator("drive_shaping_method")
    @classmethod
    def _normalize_drive_shaping_method(cls, val: Any) -> DriveType | Any:
        """Normalize the `drive_shaping_method` attribute."""
        if isinstance(val, DriveType):
            return val
        elif isinstance(val, str):
            u = val.upper()
            if u == DriveType.ADIABATIC.name:
                return DriveType.ADIABATIC
            elif u == DriveType.OPTIMIZED.name:
                return DriveType.OPTIMIZED
            else:
                raise ValueError(f"Invalid drive shaping method '{val}'.")

        elif inspect.isclass(val):
            from qubosolver.pipeline.drive import BaseDriveShaper

            if not issubclass(val, BaseDriveShaper):
                raise TypeError(f"Class must be a subclass of {BaseDriveShaper.__name__}")
            else:
                return val
        else:
            raise TypeError("Invalid drive shaping method type.")

    @field_validator("optimized_initial_omega_parameters")
    @classmethod
    def _check_optimized_initial_omega_parameters(cls, val: list[float]) -> list[float]:
        if len(val) == 3:
            return val
        else:
            raise ValueError("`optimized_initial_omega_parameters` should be a list of 3 numbers.")

    @field_validator("optimized_initial_detuning_parameters")
    @classmethod
    def _check_optimized_initial_detuning_parameters(cls, val: list[float]) -> list[float]:
        if len(val) == 3:
            return val
        else:
            raise ValueError(
                "`optimized_initial_detuning_parameters` should be a list of 3 numbers."
            )


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
        embedding (EmbeddingConfig, optional): Embedding part configuration of the solver.
        drive_shaping (DriveShapingConfig, optional): Drive-shaping part configuration
            of the solver.
        classical (ClassicalConfig, optional): Classical part configuration of the solver.
        backend (LocalEmulator | RemoteEmulator | QPU, optional): backend
            for running quantum programs. Note that parameters
            such as `dt` are directly set when creating LocalEmulator | RemoteEmulator | QPU,
            hence they are deprecated compared to previous qubo-solver versions.
            Also the number of shots is set there as well.
            Defaults to a LocalEmulator using qutip.
        device (Device, optional): The quantum device specification. Defaults to `DigitalAnalogDevice`.
        do_postprocessing (bool, optional): Whether we apply post-processing (`True`) or not (`False`).
            Defaults to True.
        do_preprocessing (bool, optional): Whether we apply pre-processing (`True`) or not (`False`).
            Defaults to True.
        activate_trivial_solutions (bool, optional): Whether calculate trivial solutions (`True`)
            or not (`False`). Defaults to True.
    """

    config_name: str = ""
    use_quantum: bool | None = False
    embedding: EmbeddingConfig = EmbeddingConfig()
    drive_shaping: DriveShapingConfig = DriveShapingConfig()
    classical: ClassicalConfig = ClassicalConfig()
    backend: LocalEmulator | RemoteEmulator | QPU = LocalEmulator(backend_type=QutipBackendV2)
    device: Device = DigitalAnalogDevice()
    do_postprocessing: bool = False
    do_preprocessing: bool = False
    activate_trivial_solutions: bool = True

    def __repr__(self) -> str:
        return self.config_name

    def specs(self) -> str:
        """Return the specs of the `SolverConfig`, that is all attributes.

        Returns:
            dict: Dictionary of specs key-values.
        """
        return "\n".join(
            f"{k}: ''" if v == "" else f"{k}: {v}" for k, v in self.model_dump().items()
        )

    def print_specs(self) -> None:
        """Print specs."""
        print(self.specs())

    @model_validator(mode="after")
    def _set_greedy_traps_greedy_spacing_from_device(self) -> SolverConfig:

        if self.device:
            device = self.device._device
            if hasattr(device, "min_layout_traps"):
                if self.embedding.greedy_traps < device.min_layout_traps:
                    self.embedding = self.embedding.model_copy(
                        update={"greedy_traps": device.value.min_layout_traps}
                    )
            if hasattr(device, "min_atom_distance"):
                greedy_spacing_device = float(device.min_atom_distance)
                if self.embedding.greedy_spacing < greedy_spacing_device:
                    self.embedding = self.embedding.model_copy(
                        update={"greedy_spacing": greedy_spacing_device}
                    )
        return self

    @classmethod
    def from_kwargs(cls, **kwargs: dict) -> SolverConfig:
        """Create an instance based on entries of other configs.

        Note that if any of the keywords
        ("embedding", "drive_shaping", "classical")
        are present in kwargs, the values are taken directly.

        Returns:
            SolverConfig: An instance from values.
        """
        # Extract fields from pydantic BaseModel
        embedding_fields = {k: v for k, v in kwargs.items() if k in EmbeddingConfig.model_fields}
        drive_shaping_fields = {
            k: v for k, v in kwargs.items() if k in DriveShapingConfig.model_fields
        }
        classical_fields = {k: v for k, v in kwargs.items() if k in ClassicalConfig.model_fields}

        solver_fields = {
            k: v
            for k, v in kwargs.items()
            if k in cls.model_fields and k not in ("embedding", "drive_shaping", "classical")
        }

        return cls(
            embedding=(
                EmbeddingConfig(**embedding_fields)
                if "embedding" not in kwargs
                else kwargs["embedding"]
            ),
            drive_shaping=(
                DriveShapingConfig(**drive_shaping_fields)
                if "drive_shaping" not in kwargs
                else kwargs["drive_shaping"]
            ),
            classical=(
                ClassicalConfig(**classical_fields)
                if "classical" not in kwargs
                else kwargs["classical"]
            ),
            **solver_fields,
        )
