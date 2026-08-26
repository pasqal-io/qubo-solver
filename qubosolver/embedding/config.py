from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field

import torch

from .enums import Algorithm, Lattice


@dataclass
class Config():
    """A module-level [`embedding.Config`][] that defines the embedding part of a [`solvers.quantum.Config`][].

    Attributes:
        algorithm: The type of embedding method used to
            place atoms on the register according to the QUBO problem.
            Defaults to `Algorithm.GREEDY_LAYOUT`.
        greedy_layout_lattice: Lattice type for the
            greedy layout embedder method. Defaults to `Lattice.TRIANGULAR`.
        greedy_layout_traps: The number of traps on the register.
            Defaults to ``"device"``, i.e. automatically set to match the selected device capacity.
            A too high value will impede computational efficiency.
        greedy_layout_max_possible_term:
            If a `float`, it corresponds to the maximum representable quadratic
            term. If a `tuple`, the first element should be `'factor'`, and the
            second element is a multiplier on the QUBO's maximum quadratic term
            to define the maximum representable quadratic term.
            Defaults to `('factor', 1.0)`. The maximum possible term corresponds
            to the interaction for the closest possible pair in the layout.
            Setting it to a higher value than the actual maximum increases the
            resolution to represent the terms. Setting it to a lower value
            decreases the resolution and allows traps to be set farther to
            potentially represent smaller terms.
        blade_steps_per_round: Maps directly to `steps_per_round` in [`qoolqit.embedding.BladeConfig`][]
        blade_starting_positions: Maps directly to `starting_positions` in [`qoolqit.embedding.BladeConfig`][]
        blade_dimensions: Maps directly to `dimensions` in [`qoolqit.embedding.BladeConfig`][]
        max_min_dist_ratio: Maximum allowed ratio
            between the largest and the smallest inter-atom distance in the resulting
            register. When ``"device"``, it is derived from the configured device's
            ``max_radial_distance`` / ``min_distance`` specs. Defaults to ``"device"``.
    """

    algorithm: Algorithm | str = Algorithm.GREEDY_LAYOUT

    greedy_layout_lattice: Lattice | str = Lattice.TRIANGULAR
    greedy_layout_traps: int | Literal["device"] = "device"
    greedy_layout_max_possible_term: float | tuple[Literal["factor"], float] = ("factor", 1.0)
    blade_steps_per_round: int | None = 200
    blade_starting_positions: torch.Tensor | None = None
    blade_dimensions: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    max_min_dist_ratio: float | Literal["device"] = "device"

    def __post_init__(self) -> None:
        self.algorithm = self._normalize_algorithm(self.algorithm)
        self.greedy_layout_lattice = self._normalize_lattice(self.greedy_layout_lattice)

    @staticmethod
    def _normalize_algorithm(val: str | Algorithm) -> Algorithm:
        """Normalize the embedded attribute."""
        if isinstance(val, Algorithm):
            return val
        elif isinstance(val, str):
            try:
                return Algorithm[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid str embedding method '{val}'.")
        else:
            raise TypeError("Invalid embedding method type.")

    @staticmethod
    def _normalize_lattice(val: str | Lattice) -> Lattice:
        """Normalize the lattice attribute."""
        if isinstance(val, Lattice):
            return val
        elif isinstance(val, str):
            try:
                return Lattice[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid lattice '{val}'.")
        else:
            raise TypeError("Invalid lattice type.")

