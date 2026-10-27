# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Embedding stage configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from qubosolver._checks import TYPE_CHECKING
from qubosolver.embedding import greedy_layout
from qubosolver.embedding.enums import Lattice

if TYPE_CHECKING:
    import qoolqit

_EmbeddingAlgorithm = Literal["blade", "greedy_layout"]
_GreedyLayoutLattice = Literal["square", "triangular"]


@dataclass
class Config:
    """A configuration that defines the embedding part of a [`QuantumSolvingConfig`][].

    Attributes:
        algorithm: The type of embedding method used to
            place atoms on the register according to the QUBO problem. One of:

            - `"blade"`: BLADE embedder using graph-theoretic optimization for qubit placement.
            - `"greedy_layout"`: Greedy layout-based embedder that places qubits on a
              regular lattice.

            Defaults to `"blade"`.
        greedy_layout_lattice: Lattice type for the
            greedy layout embedder method. One of `"square"` or `"triangular"`.
            Defaults to `"triangular"`.
        blade_steps_per_round: Maps directly to `steps_per_round` in
            [`qoolqit.embedding.BladeConfig`][]
    """

    algorithm: Literal["blade", "greedy_layout"] = "blade"

    greedy_layout_lattice: Literal["square", "triangular"] = "triangular"
    blade_steps_per_round: int | None = 200

    def __post_init__(self) -> None:
        """Validate `algorithm` and `greedy_layout_lattice`."""
        if self.algorithm not in get_args(_EmbeddingAlgorithm):
            msg = f"Invalid embedding method '{self.algorithm}'."
            raise ValueError(msg)
        if self.greedy_layout_lattice not in get_args(_GreedyLayoutLattice):
            msg = f"Invalid lattice '{self.greedy_layout_lattice}'."
            raise ValueError(msg)

    def _to_greedy_layout_config(self, device: qoolqit.Device) -> greedy_layout.Config:
        """Create a [`greedy_layout.Config`][] from this configuration.

        Maps the ``greedy_*`` fields onto the corresponding `greedy_layout.Config` attributes.

        Args:
            device: Target quantum device.

        Returns:
            A configuration fully populated from the ``greedy_*`` embedding settings.
        """
        cfg = greedy_layout.Config.from_device(device)

        match self.greedy_layout_lattice:
            case "triangular":
                cfg.lattice = Lattice.TRIANGULAR
            case "square":
                cfg.lattice = Lattice.SQUARE
            case _:
                msg = (
                    f"Unknown lattice type: {self.greedy_layout_lattice!r}. "
                    f"Expected 'triangular' or 'square'."
                )
                raise ValueError(msg)

        return cfg
