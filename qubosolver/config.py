"""Configuration classes for the QUBO solver pipeline.

This module defines dataclass-based configuration classes that control every
stage of the quantum and classical solving pipeline.

All public classes are re-exported from the top-level [`qubosolver`][] namespace
and can be imported directly:

```python
from qubosolver import (
    SolverConfig,
    ClassicalConfig,
    DecompositionConfig,
)
from qubosolver import embedding, drive_shaping
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import torch

from qoolqit import Device, AnalogDeviceWithDMM
from qoolqit.execution import QPU


from .types import (
    LocalEmulator,
    RemoteEmulator,
)
from . import embedding, drive_shaping, solvers
from .utils._config import _Config


@dataclass
class ClassicalConfig(_Config):
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

    algorithm: str | solvers.ClassicalAlgorithm = "tabu_search"

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
        val: str | solvers.ClassicalAlgorithm,
    ) -> solvers.ClassicalAlgorithm | Any:
        """Normalize the classical_solver_type attribute."""
        if isinstance(val, solvers.ClassicalAlgorithm):
            return val
        u = val.upper()
        all_names = [c.name for c in solvers.ClassicalAlgorithm]
        if u in all_names:
            return solvers.ClassicalAlgorithm[u]
        else:
            raise ValueError(f"Invalid classical algorithm '{val}'.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the fields relevant to the active solver type.

        Returns a dict containing ``classical_solver_type`` plus the subset of
        fields that are meaningful for the chosen solver
        (``CPLEX``, ``SIMULATED_ANNEALING``, or ``TABU_SEARCH``). Fields
        belonging to inactive solvers are omitted to keep serialized output
        minimal.

        Returns:
            dict[str, Any]: Serialized representation of this config.
        """
        serialization: dict = {"algorithm": self.algorithm}
        if self.algorithm == solvers.ClassicalAlgorithm.CPLEX:
            serialization.update(
                {"cplex_maxtime": self.cplex_maxtime, "cplex_log_path": self.cplex_log_path}
            )
        if self.algorithm == solvers.ClassicalAlgorithm.SIMULATED_ANNEALING:
            serialization.update(
                {
                    "max_iter": self.max_iter,
                    "max_bitstrings": self.max_bitstrings,
                    "sa_initial_temp": self.sa_initial_temp,
                    "sa_final_temp": self.sa_final_temp,
                    "sa_cooling_rate": self.sa_cooling_rate,
                    "sa_seed": self.sa_seed,
                    "sa_start": self.sa_start,
                    "sa_time_limit": self.sa_time_limit,
                }
            )
        if self.algorithm == solvers.ClassicalAlgorithm.TABU_SEARCH:
            serialization.update(
                {
                    "max_bitstrings": self.max_bitstrings,
                    "max_iter": self.max_iter,
                    "tabu_x0": self.tabu_x0,
                    "tabu_tenure": self.tabu_tenure,
                    "tabu_max_no_improve": self.tabu_max_no_improve,
                    "tabu_time_limit": self.tabu_time_limit,
                }
            )
        return serialization


@dataclass
class DecompositionConfig(_Config):
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


@dataclass
class SolverConfig(_Config):
    """
    A `SolverConfig` instance defines how a QUBO problem should be solved.
    We specify whether to use a quantum or classical approach,
    which backend to run on, and additional execution parameters.

    Attributes:
        config_name (str, optional): The name of the current configuration.
            Defaults to ''.
        use_quantum (bool, optional): Whether to solve using a quantum approach (`True`)
            or a classical approach (`False`). Defaults to True.
        embedding (embedding.Config, optional): Embedding part configuration of the solver.
        drive_shaping (drive_shaping.Config, optional): Drive-shaping part configuration
            of the solver.
        classical (ClassicalConfig, optional): Classical part configuration of the solver.
        backend (LocalEmulator | RemoteEmulator | QPU, optional): backend
            for running quantum programs. Note that parameters
            such as `dt` are directly set when creating LocalEmulator | RemoteEmulator | QPU,
            hence they are deprecated compared to previous qubo-solver versions.
            Also the number of shots is set there as well.
            Defaults to a LocalEmulator using qutip.
        device (Device, optional): The quantum device specification. Defaults to `AnalogDeviceWithDMM`.
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
    use_quantum: bool = True
    embedding: embedding.Config = field(default_factory=embedding.Config)
    drive_shaping: drive_shaping.Config = field(default_factory=drive_shaping.Config)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    backend: LocalEmulator | RemoteEmulator | QPU = field(default_factory=LocalEmulator)
    device: Device = field(default_factory=AnalogDeviceWithDMM)
    do_postprocessing: bool = False
    do_preprocessing: bool = False
    activate_trivial_solutions: bool = True
    decompose: DecompositionConfig | None = None

    def __repr__(self) -> str:
        return self.config_name

    def specs(self) -> str:
        """Return a human-readable summary of all configuration attributes.

        Each attribute is formatted as ``key: value``, one per line.
        Empty-string values are rendered as ``key: ''``.

        Returns:
            str: Newline-separated ``key: value`` pairs for all config fields.
        """
        return "\n".join(f"{k}: ''" if v == "" else f"{k}: {v}" for k, v in self.to_dict().items())

    def print_specs(self) -> None:
        """Print all configuration attributes to stdout.

        Convenience wrapper around :meth:`specs` for interactive use.
        """
        print(self.specs())

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

    @classmethod
    def from_kwargs(cls, **kwargs: dict) -> SolverConfig:
        """Create a ``SolverConfig`` from a flat or mixed keyword dictionary.

        Keyword arguments are automatically routed to the appropriate
        sub-config (``embedding.Config``, ``drive_shaping.Config``,
        ``ClassicalConfig``, or ``DecompositionConfig``) based on their field
        names. Top-level ``SolverConfig`` fields are handled directly.

        If any of the sub-config keys (``"embedding"``, ``"drive_shaping"``,
        ``"classical"``, ``"decompose"``) appear in ``kwargs``, their values
        are forwarded as-is and take precedence over individually routed fields.

        Args:
            **kwargs: Any combination of fields from ``SolverConfig``,
                ``embedding.Config``, ``drive_shaping.Config``,
                ``ClassicalConfig``, or ``DecompositionConfig``.

        Returns:
            SolverConfig: A fully validated ``SolverConfig`` instance.
        """

        def _validate(config_cls: type, value: Any) -> Any:
            """Build `config_cls` from a dict, or pass through an existing instance."""
            return value if isinstance(value, config_cls) else config_cls(**value)

        embedding_fields = {k: v for k, v in kwargs.items() if k in embedding.Config.field_names()}
        drive_shaping_fields = {
            k: v for k, v in kwargs.items() if k in drive_shaping.Config.field_names()
        }
        classical_fields = {k: v for k, v in kwargs.items() if k in ClassicalConfig.field_names()}
        decompose_fields = {
            k: v for k, v in kwargs.items() if k in DecompositionConfig.field_names()
        } or kwargs.get("decompose", {})

        solver_fields: dict[str, Any] = {
            k: v
            for k, v in kwargs.items()
            if k in cls.field_names()
            and k not in ("embedding", "drive_shaping", "classical", "decompose")
        }
        solver_fields["embedding"] = _validate(
            embedding.Config, kwargs.get("embedding", embedding_fields)
        )
        solver_fields["drive_shaping"] = _validate(
            drive_shaping.Config, kwargs.get("drive_shaping", drive_shaping_fields)
        )
        solver_fields["classical"] = _validate(
            ClassicalConfig, kwargs.get("classical", classical_fields)
        )
        if decompose_fields:
            solver_fields["decompose"] = _validate(DecompositionConfig, decompose_fields)

        return cls(**solver_fields)
