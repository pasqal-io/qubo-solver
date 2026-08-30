from __future__ import annotations

from dataclasses import dataclass, field
import torch
from typing import Literal, get_args

import qoolqit

from .embedding import Config as EmbeddingConfig
from .drive_shaping import Config as DriveShapingConfig
from qubosolver.types.backends import LocalEmulator, RemoteEmulator



ClassicalAlgorithm = Literal["tabu_search", "simulated_annealing", "cplex", "random_sampling"]


@dataclass
class ClassicalConfig():
    """A `classical.Config` instance defines the classical part of a `SolverConfig`.

    Attributes:
        algorithm (ClassicalAlgorithm, optional): Classical solver type. One of:

            - `"tabu_search"`: Tabu search metaheuristic that avoids recently visited solutions.
            - `"simulated_annealing"`: Simulated annealing algorithm that probabilistically
              accepts worse solutions to escape local minima.
            - `"cplex"`: IBM CPLEX exact solver; requires a valid CPLEX installation and licence.
            - `"random_sampling"`: Randomly samples solutions; useful as a baseline or for testing.

            Defaults to `"tabu_search"`.
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

    algorithm: ClassicalAlgorithm = "tabu_search"

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
        if self.algorithm not in get_args(ClassicalAlgorithm):
            raise ValueError(f"Invalid classical algorithm '{self.algorithm}'.")


@dataclass
class QuantumConfig():
    """A `quantum.Config` instance defines the quantum part of a `SolverConfig`.

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

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    drive_shaping: DriveShapingConfig = field(default_factory=DriveShapingConfig)
    backend: LocalEmulator | RemoteEmulator | qoolqit.execution.QPU = field(default_factory=LocalEmulator)
    device: qoolqit.Device = field(default_factory=qoolqit.AnalogDeviceWithDMM)

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
