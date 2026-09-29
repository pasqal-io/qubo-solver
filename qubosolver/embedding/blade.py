"""BLaDE (Balanced Layout and Distance Embedding) adapter for QUBO instances.

This module is a thin wrapper around [`qoolqit.embedding.Blade`][] that
exposes a single [`embed`] entry point accepting an
[`Instance`][] and returning a [`qoolqit.Register`][] ready for use
in a quantum program.

BLaDE maps the QUBO coefficient matrix onto a 2-D (or higher-dimensional)
set of atom positions so that the physical interaction strengths
(∝ 1/‖rᵢ - rⱼ‖⁶) are as proportional to the QUBO edge weights as possible.
It does so by iteratively refining coordinates across multiple dimensional
reduction rounds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import qoolqit
from qoolqit.embedding import Blade, BladeConfig

from qubosolver import Instance, tensor
from qubosolver.embedding._algorithms.mds import embed_mds
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal

logger = logging.getLogger(__name__)


def _constant_weight_relative_threshold(_: float) -> float:
    return 0.1


def _constant_regulation_cursor(_: float) -> float:
    return 0.5


@dataclass
class Config(BladeConfig):
    """[`qoolqit.embedding.BladeConfig`][] with an optional MDS initialization.

    Its defaults were tuned on a benchmark of native QUBO instances: BLaDE starts
    from an MDS of the QUBO and explores the dimensions `(6, 5, 4, 3, 2, 2, 2)`,
    with a constant weight relative threshold of `0.1` and a constant regulation
    cursor of `0.5`.

    Attributes:
        initialize_with_mds: Whether BLaDE starts from a multi-dimensional scaling
            (MDS) of the QUBO. MDS is skipped when `starting_positions` is set, and
            for instances without positive off-diagonal coefficient. Defaults to `True`.

    Warning:
        `max_min_dist_ratio` is an advanced parameter: set it manually (or via
        the `device` constructor argument) with care, since a bad value can
        produce a register that the target device cannot realize.
    """

    dimensions: tuple[int, ...] = (6, 5, 4, 3, 2, 2, 2)
    compute_weight_relative_threshold: Callable[[float], float] = (
        _constant_weight_relative_threshold
    )
    compute_regulation_cursor: Callable[[float], float] = _constant_regulation_cursor
    initialize_with_mds: bool = True

    def _to_qoolqit(self) -> BladeConfig:
        """Return the equivalent [`qoolqit.embedding.BladeConfig`][].

        qoolqit passes every config field to BLaDE, so `initialize_with_mds` is dropped.
        """
        return BladeConfig(**{f.name: getattr(self, f.name) for f in fields(BladeConfig)})


def embed(
    instance: Instance,
    *,
    config: Config | None = None,
) -> qoolqit.Register:
    """Embed a QUBO instance using the BLaDE algorithm.

    Runs the BLaDE optimization on the QUBO coefficient matrix. Atom
    labels are assigned as integer indices (``0``, ``1``, …)
    matching the variable ordering of the QUBO matrix.

    Warning:
        A poorly chosen `config` can produce a register that is incompatible
        with a target device. See [`Config`][].

    Args:
        instance: The QUBO instance to embed.
        config: BLaDE configuration controlling the optimization (number of
            steps per round, initial atom positions, dimension sequence,
            maximum allowed ratio of radial to minimum distance, etc.).

    Returns:
        A register mapping each atom label to its 2-D position, with atom positions
            determined by BLaDE.

    Raises:
        ValueError: If `instance` has no variables (``size == 0``), since a
            register must contain at least one qubit.
        ValueError: If the QUBO coefficient matrix has negative off-diagonal
            coefficients, since BLaDE cannot embed such instances.
    """
    config = config or Config()
    logger.debug("embed: instance size=%d, config=%r", instance.size, config)
    if not instance:
        raise ValueError("Cannot embed an empty instance (size=0): nothing to place.")

    if _has_negative_offdiagonal(instance.matrix):
        raise ValueError("QUBOs with negative off-diagonal coefficients cannot be embedded.")

    if instance.size == 1:
        # A single atom has no off-diagonal term to place it relative to,
        # so it is placed at the origin without running the algorithm.
        return qoolqit.Register.from_coordinates(tensor.zeros(1, 2))

    qubo = instance.matrix.numpy()

    blade_config = config._to_qoolqit()
    if config.initialize_with_mds:
        if blade_config.starting_positions is not None:
            logger.info("`starting_positions` is set: skipping the MDS initialization.")
        elif (np.triu(qubo, k=1) > 0).any():
            blade_config.starting_positions = embed_mds(qubo)

    _blade = Blade(blade_config)
    graph = _blade.embed(qubo)
    register = qoolqit.Register.from_graph(graph)
    return register


def embed_for_device(
    instance: Instance,
    device: qoolqit.Device,
) -> qoolqit.Register:
    """Embed a QUBO instance using the BLaDE algorithm, sized for *device*.

    Convenience wrapper around `embed` that derives `Config.max_min_dist_ratio`
    from *device* via `Config`'s `device` constructor argument.

    Args:
        instance: The QUBO instance to embed.
        device: Target quantum device the resulting register must fit.

    Returns:
        A register mapping each atom label to its 2-D position, with atom positions
            determined by BLaDE.

    To also tune device-independent parameters (e.g. `dimensions` or
    `steps_per_round`), pass them directly to `Config` alongside `device`:

    Example:
        ```python
        config = Config(device=device, steps_per_round=100)
        register = embed(instance, config=config)
        ```
    """
    logger.debug("embed_for_device: instance size=%d, device=%r", instance.size, device)
    return embed(instance, config=Config(device=device))
