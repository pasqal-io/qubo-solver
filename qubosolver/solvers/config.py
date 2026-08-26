from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import classical, quantum

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
        solving (quantum.Config | classical.Config, optional): Whether to solve using a quantum
            approach (`quantum.Config`) or a classical approach (`classical.Config`), together
            with the configuration of that approach. Defaults to a `quantum.Config`.
        do_postprocessing (bool, optional): Whether we apply post-processing (`True`) or not (`False`).
            Defaults to True.
        do_preprocessing (bool, optional): Whether we apply pre-processing (`True`) or not (`False`).
            Defaults to True.
        activate_trivial_solutions (bool, optional): Whether calculate trivial solutions (`True`)
            or not (`False`). Defaults to True.
        decompose (DecompositionConfig | None, optional): which decomposition configuration to use
            when solving large QUBOs. Defaults to None, i.e. no decomposition is applied.
        postprocessing_time_limit (float, optional): Maximum total time in seconds for the
            whole post-processing batch, shared across all bitstrings. Defaults to
            `float("inf")`, meaning no time limit.
    """

    config_name: str = ""
    solving: quantum.Config | classical.Config = field(default_factory=quantum.Config)
    do_postprocessing: bool = False
    do_preprocessing: bool = False
    activate_trivial_solutions: bool = True
    decompose: DecompositionConfig | None = None
    postprocessing_time_limit: float = float("inf")

    def __repr__(self) -> str:
        return self.config_name

    @property
    def solving_mode(self) -> Literal["quantum", "classical"]:
        match self.solving:
            case quantum.Config():
                return "quantum"
            case classical.Config():
                return "classical"
            case _:
                raise ValueError(f"Invalid solving config '{self.solving!r}'.")

    @property
    def quantum(self) -> quantum.Config:
        if self.solving_mode != "quantum":
            raise ValueError(f"Config '{self.config_name}' is not configured for quantum solving.")
        assert isinstance(self.solving, quantum.Config)
        return self.solving

    @property
    def classical(self) -> classical.Config:
        if self.solving_mode != "classical":
            raise ValueError(f"Config '{self.config_name}' is not configured for classical solving.")
        assert isinstance(self.solving, classical.Config)
        return self.solving
