from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .solving import ClassicalConfig as ClassicalSolvingConfig
from .solving import QuantumConfig as QuantumSolvingConfig


@dataclass
class DecompositionConfig:
    """The configuration parameters when using a decomposition method
    for solving large QUBO instances.

    Attributes:
        decompose_threshold: Threshold value for cost function used
            when searching to place a node/variable during decomposition.
        decompose_stop_number: Maximal number of nodes/variables left
            after the decomposition loop.
        decompose_break_placement: If a search iteration ends with very
            few nodes to place/variables on device, we stop iterating.
        neglecting_inter_distance: Value for neglecting interactions in the
            distance interaction matrix.
        neglecting_max_coefficient: QUBO coefficient from which we consider
            an interaction is neglecting.
        classical_max_min_dist_ratio: Maximal ratio between the largest and
            smallest distances allowed before falling back to a classical
            approach. Defaults to `float("inf")`, meaning no limit.
    """

    decompose_threshold: float = 250.0
    decompose_stop_number: int = 15
    decompose_break_placement: int = 3
    neglecting_inter_distance: float = 1.5
    neglecting_max_coefficient: float = 1.0
    classical_max_min_dist_ratio: float = float("inf")


@dataclass
class SolverConfig:
    """A configuration instance that defines how a QUBO problem should be solved.

    We specify whether to use a quantum or classical approach, which backend
    to run on, and additional execution parameters.

    Attributes:
        config_name: The name of the current configuration.
            Defaults to `""`.
        solving: Whether to solve using a quantum approach ([`QuantumSolvingConfig`][])
            or a classical approach ([`ClassicalSolvingConfig`][]), together with
            the configuration of that approach. Defaults to a `QuantumSolvingConfig`.
        postprocessing: Whether we apply post-processing (`True`) or not
            (`False`). Defaults to `False`.
        preprocessing: Whether we apply pre-processing (`True`) or not
            (`False`). Defaults to `False`.
        activate_trivial_solutions: Whether to calculate trivial solutions
            (`True`) or not (`False`). Defaults to `True`.
        decompose: Which decomposition configuration to use when solving
            large QUBOs. Defaults to `None`, i.e. no decomposition is applied.
        postprocessing_time_limit: Maximum total time in seconds for the
            whole post-processing batch, shared across all bitstrings. Defaults
            to `float("inf")`, meaning no time limit.
    """

    config_name: str = ""
    solving: QuantumSolvingConfig | ClassicalSolvingConfig = field(
        default_factory=QuantumSolvingConfig
    )
    postprocessing: bool = False
    preprocessing: bool = False
    activate_trivial_solutions: bool = True
    decompose: DecompositionConfig | None = None
    postprocessing_time_limit: float = float("inf")

    def __repr__(self) -> str:
        return self.config_name

    @property
    def solving_mode(self) -> Literal["quantum", "classical"]:
        """
        Returns:
            `"quantum"` if [`solving`][] is a [`QuantumSolvingConfig`][], or
                `"classical"` if it is a [`ClassicalSolvingConfig`][].

        Raises:
            ValueError: If `solving` is neither a [`QuantumSolvingConfig`][] nor a
                [`ClassicalSolvingConfig`][].
        """
        match self.solving:
            case QuantumSolvingConfig():
                return "quantum"
            case ClassicalSolvingConfig():
                return "classical"
            case _:
                raise ValueError(f"Invalid solving config '{self.solving!r}'.")

    @property
    def quantum(self) -> QuantumSolvingConfig:
        """Access the quantum solving configuration directly, without checking
        [`solving_mode`][] yourself — this also lets type-checkers narrow the type
        without an explicit [`isinstance`][] check or [`cast`][typing.cast] at the call site.

        Returns:
            The quantum solving configuration, if in quantum solving mode.

        Raises:
            ValueError: If this configuration is not configured for quantum solving.
        """
        if self.solving_mode != "quantum":
            raise ValueError(f"Config '{self.config_name}' is not configured for quantum solving.")
        assert isinstance(self.solving, QuantumSolvingConfig)
        return self.solving

    @property
    def classical(self) -> ClassicalSolvingConfig:
        """Access the classical solving configuration directly, without checking
        [`solving_mode`][] yourself — this also lets type-checkers narrow the type
        without an explicit [`isinstance`][] check or [`cast`][typing.cast] at the call site.

        Returns:
            The classical solving configuration, if in classical solving mode.

        Raises:
            ValueError: If this configuration is not configured for classical solving.
        """
        if self.solving_mode != "classical":
            raise ValueError(
                f"Config '{self.config_name}' is not configured for classical solving."
            )
        assert isinstance(self.solving, ClassicalSolvingConfig)
        return self.solving
