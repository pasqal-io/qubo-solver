"""Solving stage configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import qoolqit
import torch

from qubosolver.types.backends import LocalEmulator, RemoteEmulator

from .drive_shaping import Config as DriveShapingConfig
from .embedding import Config as EmbeddingConfig

_ClassicalAlgorithm = Literal["tabu_search", "simulated_annealing", "cplex", "random_sampling"]


@dataclass
class ClassicalConfig:
    """A configuration that defines the classical-solving part of a [`SolverConfig`][].

    Attributes:
        algorithm: Classical solver algorithm. One of:

            - `"tabu_search"`: Tabu search metaheuristic that avoids recently visited solutions.
            - `"simulated_annealing"`: Simulated annealing algorithm that probabilistically
              accepts worse solutions to escape local minima.
            - `"cplex"`: IBM CPLEX exact solver; requires a valid CPLEX installation and license.
            - `"random_sampling"`: Randomly samples solutions; useful as a baseline or for testing.

            Defaults to `"tabu_search"`.
        time_limit: Maximum runtime in seconds for the classical solve (cplex, simulated
            annealing, or tabu search). Defaults to `float("inf")`, meaning no time limit.
        max_iter: Maximum number of iterations to perform for simulated annealing or tabu search.
        max_bitstrings: Maximal number of bitstrings returned as solutions.
    """

    algorithm: Literal["tabu_search", "simulated_annealing", "cplex", "random_sampling"] = (
        "tabu_search"
    )

    time_limit: float = float("inf")

    max_iter: int = 100
    max_bitstrings: int = 1

    def __post_init__(self) -> None:
        """Validate `algorithm`."""
        if self.algorithm not in get_args(_ClassicalAlgorithm):
            raise ValueError(f"Invalid classical algorithm '{self.algorithm}'.")


@dataclass
class QuantumConfig:
    """A configuration defines the quantum-solving part of a [`SolverConfig`][].

    Attributes:
        embedding: Embedding part configuration of the solver.
        drive_shaping: Drive-shaping part configuration
            of the solver.
        backend: backend for running quantum programs. Defaults to a [`LocalEmulator`][].
        device: The quantum device specification. Defaults to [`qoolqit.AnalogDeviceWithDMM`][].
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    drive_shaping: DriveShapingConfig = field(default_factory=DriveShapingConfig)
    backend: LocalEmulator | RemoteEmulator | qoolqit.execution.QPU = field(
        default_factory=LocalEmulator
    )
    device: qoolqit.Device = field(default_factory=qoolqit.AnalogDeviceWithDMM)

    @property
    def max_min_dist_ratio(self) -> float:
        """Maximum allowed ratio between the largest and smallest inter-atom distance.

        Derived from the configured device's ``max_radial_distance`` / ``min_distance``
        specs (or ``inf`` when the device imposes no such limits).

        Returns:
            The resolved maximum min/max distance ratio.
        """
        specs = self.device.specs
        min_distance = specs["min_distance"]
        max_radial_distance = specs["max_radial_distance"]
        if min_distance is not None and min_distance > 0 and max_radial_distance is not None:
            return max_radial_distance / min_distance
        return torch.inf
