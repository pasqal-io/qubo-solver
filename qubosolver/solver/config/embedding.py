from __future__ import annotations

from typing import Literal, get_args
from dataclasses import dataclass, field

import torch

EmbeddingAlgorithm = Literal["greedy_layout", "blade"]
GreedyLayoutLattice = Literal["square", "triangular"]


@dataclass
class Config():
    """A module-level [`embedding.Config`][] that defines the embedding part of a [`solvers.quantum.Config`][].

    Attributes:
        algorithm (EmbeddingAlgorithm, optional): The type of embedding method used to
            place atoms on the register according to the QUBO problem. One of:

            - `"greedy_layout"`: Greedy layout-based embedder that places qubits on a
              regular lattice.
            - `"blade"`: BLADE embedder using graph-theoretic optimization for qubit placement.

            Defaults to `"greedy_layout"`.
        greedy_layout_lattice (GreedyLayoutLattice, optional): Lattice type for the
            greedy layout embedder method. One of `"square"` or `"triangular"`.
            Defaults to `"triangular"`.
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

    algorithm: EmbeddingAlgorithm = "greedy_layout"

    greedy_layout_lattice: GreedyLayoutLattice = "triangular"
    greedy_layout_traps: int | Literal["device"] = "device"
    greedy_layout_max_possible_term: float | tuple[Literal["factor"], float] = ("factor", 1.0)
    blade_steps_per_round: int | None = 200
    blade_starting_positions: torch.Tensor | None = None
    blade_dimensions: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    max_min_dist_ratio: float | Literal["device"] = "device"

    def __post_init__(self) -> None:
        if self.algorithm not in get_args(EmbeddingAlgorithm):
            raise ValueError(f"Invalid embedding method '{self.algorithm}'.")
        if self.greedy_layout_lattice not in get_args(GreedyLayoutLattice):
            raise ValueError(f"Invalid lattice '{self.greedy_layout_lattice}'.")

