from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from qoolqit import Device, AnalogDeviceWithDMM
from qoolqit.execution import QPU

from qubosolver.types import LocalEmulator, RemoteEmulator
from qubosolver import embedding, drive_shaping
from . import ClassicalAlgorithm


@dataclass
class ClassicalConfig():
    """A `ClassicalConfig` instance defines the classical part of a `SolverConfig`.

    Attributes:
        classical_solver_type (solvers.ClassicalAlgorithm, optional): Classical solver type. Defaults to
            `"tabu_search"`.
        cplex_maxtime (float, optional): CPLEX maximum runtime in seconds. Defaults to 600s.
        cplex_log_path (str, optional): CPLEX log path. Default to `solver.log`.
        max_iter (int, optional): Maximum number of iterations to perform for simulated annealing or tabu search.
        max_bitstrings (int, optional): Maximal number of bitstrings returned as solutions.
        sa_initial_temp (float, optional): Starting temperature (controls exploration).
        sa_final_temp (float, optional): Minimum temperature threshold for stopping.
        sa_cooling_rate (float, optional): Cooling rate - should be slightly below 1 (e.g., 0.95–0.99).
        sa_seed (int, optional): Random seed for reproducibility.
        sa_start (torch.Tensor | None, optional): Optional initial bitstring of shape (n,).
        sa_time_limit (float): Maximum runtime in seconds for simulated annealing.
            Defaults to `float("inf")`, meaning no time limit.
        tabu_x0 (torch.Tensor | None, optional): The initial binary solution tensor of shape (n,).
        tabu_tenure (int, optional): Number of iterations a move (bit flip) remains tabu.
        tabu_max_no_improve (int, optional): Maximum number of consecutive iterations
            without improvement before termination.
        tabu_time_limit (float): Maximum execution time for tabu search,
            in seconds. Defaults to `float("inf")`, meaning no time limit.
    """

    algorithm: str | ClassicalAlgorithm = "tabu_search"

    cplex_maxtime: float = 600.0
    cplex_log_path: str = ""

    max_iter: int = 100
    max_bitstrings: int = 1

    sa_initial_temp: float = 10.0
    sa_final_temp: float = 0.1
    sa_cooling_rate: float | None = None
    sa_seed: int | None = None
    sa_start: torch.Tensor | None = None
    sa_time_limit: float = float("inf")

    tabu_x0: torch.Tensor | None = None
    tabu_tenure: int = 7
    tabu_max_no_improve: int = 20
    tabu_time_limit: float = float("inf")

    def __post_init__(self) -> None:
        self.algorithm = self._normalize_classical_solver_type(self.algorithm)

    @staticmethod
    def _normalize_classical_solver_type(
        val: str | ClassicalAlgorithm,
    ) -> ClassicalAlgorithm:
        """Normalize the classical_solver_type attribute."""
        if isinstance(val, ClassicalAlgorithm):
            return val
        elif isinstance(val, str):
            try:
                return ClassicalAlgorithm[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid classical algorithm '{val}'.")
        else:
            raise TypeError("Invalid classical algorithm type.")

@dataclass
class QuantumConfig():
    """A `QuantumConfig` instance defines the quantum part of a `SolverConfig`.

    Attributes:
        embedding (embedding.Config, optional): Embedding part configuration of the solver.
        drive_shaping (drive_shaping.Config, optional): Drive-shaping part configuration
            of the solver.
        backend (LocalEmulator | RemoteEmulator | QPU, optional): backend
            for running quantum programs. Note that parameters
            such as `dt` are directly set when creating LocalEmulator | RemoteEmulator | QPU,
            hence they are deprecated compared to previous qubo-solver versions.
            Also the number of shots is set there as well.
            Defaults to a LocalEmulator using qutip.
        device (Device, optional): The quantum device specification. Defaults to `AnalogDeviceWithDMM`.
    """

    embedding: embedding.Config = field(default_factory=embedding.Config)
    drive_shaping: drive_shaping.Config = field(default_factory=drive_shaping.Config)
    backend: LocalEmulator | RemoteEmulator | QPU = field(default_factory=LocalEmulator)
    device: Device = field(default_factory=AnalogDeviceWithDMM)

    @property
    def max_min_dist_ratio(self) -> float:
        """Maximum allowed ratio between the largest and smallest inter-atom distance.

        Resolves ``embedding.max_min_dist_ratio``: returns it directly unless it is
        the sentinel ``"device"``, in which case the ratio is derived from the
        configured device's ``max_radial_distance`` / ``min_distance`` specs
        (or ``inf`` when the device imposes no such limits).

        Returns:
            float: The resolved maximum min/max distance ratio.
        """
        if self.embedding.max_min_dist_ratio != "device":
            return self.embedding.max_min_dist_ratio
        specs = self.device.specs
        min_distance = specs["min_distance"]
        max_radial_distance = specs["max_radial_distance"]
        if min_distance is not None and min_distance > 0 and max_radial_distance is not None:
            return max_radial_distance / min_distance
        return torch.inf


@dataclass
class DecompositionConfig():
    """The configuration parameters when using a decomposition method
        for solving large QUBO instances.

    Attributes:
        decompose_threshold (float, optional): Threshold value for cost function used
            when searching to place a node/variable during decomposition.
        decompose_stop_number (int, optional): Maximal number of nodes/variables left
            after the decomposition loop.
        decompose_break_placement (int, optional): If a search iteration ends with very
            few nodes to place/variables on device, we stop iterating.
        neglecting_inter_distance (float, optional): Value
            for neglecting interactions in the distance interaction matrix.
        neglecting_max_coefficient (float, optional): Qubo coefficient from which
            we consider an interaction is neglecting.
    """

    decompose_threshold: float = 250.0
    decompose_stop_number: int = 15
    decompose_break_placement: int = 3
    neglecting_inter_distance: float = 1.5
    neglecting_max_coefficient: float = 1.0
    classical_max_min_dist_ratio: float = float("inf")


@dataclass
class Config():
    """
    A `SolverConfig` instance defines how a QUBO problem should be solved.
    We specify whether to use a quantum or classical approach,
    which backend to run on, and additional execution parameters.

    Attributes:
        config_name (str, optional): The name of the current configuration.
            Defaults to ''.
        solving (QuantumConfig | ClassicalConfig, optional): Whether to solve using a quantum
            approach (`QuantumConfig`) or a classical approach (`ClassicalConfig`), together
            with the configuration of that approach. Defaults to a `QuantumConfig`.
        do_postprocessing (bool, optional): Whether we apply post-processing (`True`) or not (`False`).
            Defaults to True.
        do_preprocessing (bool, optional): Whether we apply pre-processing (`True`) or not (`False`).
            Defaults to True.
        activate_trivial_solutions (bool, optional): Whether calculate trivial solutions (`True`)
            or not (`False`). Defaults to True.
        decompose (DecompositionConfig | None, optional): which decomposition configuration to use
            when solving large QUBOs. Defaults to None, i.e. no decomposition is applied.
    """

    config_name: str = ""
    solving: QuantumConfig | ClassicalConfig = field(default_factory=QuantumConfig)
    do_postprocessing: bool = False
    do_preprocessing: bool = False
    activate_trivial_solutions: bool = True
    decompose: DecompositionConfig | None = None

    def __repr__(self) -> str:
        return self.config_name

    @property
    def solving_mode(self) -> Literal["quantum", "classical"]:
        match self.solving:
            case QuantumConfig():
                return "quantum"
            case ClassicalConfig():
                return "classical"
            case _:
                raise ValueError(f"Invalid solving config '{self.solving!r}'.")

    @property
    def quantum(self) -> QuantumConfig:
        if self.solving_mode != "quantum":
            raise ValueError(f"Config '{self.config_name}' is not configured for quantum solving.")
        assert isinstance(self.solving, QuantumConfig)
        return self.solving

    @property
    def classical(self) -> ClassicalConfig:
        if self.solving_mode != "classical":
            raise ValueError(f"Config '{self.config_name}' is not configured for classical solving.")
        assert isinstance(self.solving, ClassicalConfig)
        return self.solving
