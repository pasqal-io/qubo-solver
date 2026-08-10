"""BLaDE (Balanced Layout and Distance Embedding) adapter for QUBO instances.

This module is a thin wrapper around `qoolqit.embedding.Blade` that
exposes a single `embed` entry point accepting a
[`qubosolver.Instance`][] and returning a `qoolqit.Register` ready for use
in a quantum program.

BLaDE maps the QUBO coefficient matrix onto a 2-D (or higher-dimensional)
set of atom positions so that the physical interaction strengths
(∝ 1/‖rᵢ − rⱼ‖⁶) are as proportional to the QUBO edge weights as possible.
It does so by iteratively refining coordinates across multiple dimensional
reduction rounds.

Typical usage goes through [`qubosolver.embedding.blade.embed`][], which
reads [`qubosolver.embedding.blade.Config`][] parameters.
"""

from __future__ import annotations

from typing import TypeAlias

from qubosolver import Instance, tensor
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal
from qoolqit import Register
from qoolqit.embedding import Blade, BladeConfig

# Alias BladeConfig under the module-local name ``Config`` so callers can
# refer to ``blade.Config`` without importing from qoolqit directly.
# TODO: Replace TypeAlias with the ``type`` statement when Python >= 3.12.
Config: TypeAlias = BladeConfig
"""Alias for `qoolqit.BladeConfig`."""


def embed(
    instance: Instance,
    *,
    config: Config = Config(),
) -> Register:
    """Embed a QUBO instance using the BLaDE algorithm.

    Runs the BLaDE optimisation on the QUBO coefficient matrix and converts
    the resulting graph coordinates into a `qoolqit.Register`.  Atom
    labels are assigned as stringified integer indices (``"0"``, ``"1"``, …)
    matching the variable ordering of the QUBO matrix.

    Args:
        instance: The QUBO instance to embed.
        config: BLaDE configuration controlling the optimisation (number of
            steps per round, initial atom positions, dimension sequence,
            maximum allowed ratio of radial to minimum distance, etc.).

    Returns:
        A register mapping each atom label to its 2-D position, with atom positions determined by BLaDE.

    Raises:
        ValueError: If *instance* has no variables (``size == 0``), since a
            register must contain at least one qubit.
    """
    if not instance:
        raise ValueError("Cannot embed an empty instance (size=0): nothing to place.")

    if _has_negative_offdiagonal(instance.matrix):
        raise ValueError("QUBOs with negative off-diagonal coefficients cannot be embedded.")

    if instance.size == 1:
        # A single atom has no off-diagonal term to place it relative to,
        # so it is placed at the origin without running the algorithm.
        return Register.from_coordinates(tensor.zeros(1, 2))

    _blade = Blade(config)
    graph = _blade.embed(instance.matrix.numpy())

    register = Register({str(i): coord for (i, coord) in enumerate(graph.coords.values())})

    return register
