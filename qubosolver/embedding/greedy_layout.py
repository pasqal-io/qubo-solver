"""Greedy layout-based embedding algorithm for QUBO instances.

The greedy algorithm places logical QUBO nodes one at a time onto trap sites
of a pre-defined lattice (triangular or square), choosing at each step the
(node, trap) pair that minimizes the incremental mismatch between the QUBO
coefficient matrix and the physical interaction matrix (∝ 1/‖rᵢ - rⱼ‖⁶).
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import qoolqit

from qubosolver import Instance, tensor
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal
from qubosolver.utils.quantum import _max_min_distance_ratio

from ._algorithms import greedy
from .enums import Lattice, Norm

if TYPE_CHECKING:
    from qubosolver import EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the greedy layout embedding algorithm.

    Use `Config.from_device` to derive `traps` and `max_min_dist_ratio`
    from a device's constraints instead of setting them by hand.

    Warning:
        `traps` and `max_min_dist_ratio` are advanced parameters: set them
        manually (or overwrite them after calling `from_device`) with care,
        since a bad value can produce a register that the target device
        cannot realize.

    Attributes:
        traps: Number of trap sites in the layout.
        max_possible_term: Largest QUBO interaction term representable at the
            minimum trap-trap distance, in adimensional units. One of:

            - ``('quantile', q)``: the ``q`` quantile (in ``[0, 1]``) of the QUBO
              instance's strictly positive off-diagonal coefficients.
            - ``('factor', f)``: ``f`` times the QUBO instance's largest
              off-diagonal coefficient.
            - A float, used directly.

            The corresponding spacing is ``max_possible_term ** (-1 / 6)``, since
            interactions scale as ``1 / distance ** 6``.
        lattice: Lattice pattern (square or triangular).
        max_min_dist_ratio: Maximum allowed ratio between the largest and
            the smallest inter-atom distance in the resulting register.
        norm: Norm minimized when matching the QUBO coefficients to the
            physical interactions, `‖U - Q‖`. `Norm.L1` sums absolute
            deviations and spreads the error evenly; `Norm.L2` penalizes large
            individual errors more. Defaults to `Norm.L1`.
    """

    traps: int = 200
    max_min_dist_ratio: float = float("inf")
    max_possible_term: tuple[Literal["quantile", "factor"], float] | float = ("quantile", 0.95)
    lattice: Lattice = Lattice.TRIANGULAR
    norm: Norm = Norm.L1

    def __post_init__(self) -> None:
        """Initialize the private animation-related attributes."""
        self._draw_steps: bool = False
        self._animation_save_path: pathlib.Path | None = None

    @staticmethod
    def from_device(device: qoolqit.Device) -> Config:
        """Create a [`Config`][] with `traps` and `max_min_dist_ratio` derived from *device*.

        Use this to size the embedding to what *device* actually supports.
        All fields can be overwritten on the returned instance.

        Args:
            device: Target quantum device to derive the layout constraints from.

        Returns:
            A configuration with device-derived `traps` and `max_min_dist_ratio`.
        """
        return Config(
            traps=_number_of_traps_from_device(device),
            max_min_dist_ratio=_max_min_distance_ratio(device),
        )

    @staticmethod
    def _from_embedding_config(config: EmbeddingConfig, device: qoolqit.Device) -> Config:
        """Create a [`Config`][] from a user-facing [`EmbeddingConfig`][].

        Maps the ``greedy_*`` fields of *config* onto the corresponding
        `Config` attributes.

        Args:
            config: The embedding configuration to convert.
            device: Target quantum device.

        Returns:
            A configuration fully populated from the ``greedy_*`` embedding settings of *config*.
        """
        cfg = Config.from_device(device)

        match config.greedy_layout_lattice:
            case "triangular":
                cfg.lattice = Lattice.TRIANGULAR
            case "square":
                cfg.lattice = Lattice.SQUARE
            case _:
                raise ValueError(
                    f"Unknown lattice type: {config.greedy_layout_lattice!r}. "
                    f"Expected 'triangular' or 'square'."
                )

        return cfg


def _resolve_max_possible_term(
    max_possible_term: tuple[Literal["quantile", "factor"], float] | float, instance: Instance
) -> float:
    """Resolve a `Config.max_possible_term` value to a plain float.

    Args:
        max_possible_term: One of:

            - ``('quantile', q)``: resolved as the ``q`` quantile (in ``[0, 1]``)
              of *instance*'s strictly positive off-diagonal coefficients.
            - ``('factor', f)``: resolved as ``f`` times *instance*'s largest
              off-diagonal coefficient.
            - A float, returned as-is.
        instance: The QUBO instance being embedded, used to resolve the
            tuple forms.

    Returns:
        The resolved maximum representable quadratic term, as a float.

    Raises:
        ValueError: If *max_possible_term* is a tuple whose first element is
            neither ``'quantile'`` nor ``'factor'``, if *instance* has no
            strictly positive off-diagonal coefficient to resolve it from, or
            if the resolved value is not strictly positive.
    """
    if isinstance(max_possible_term, float):
        return _check_positive_max_possible_term(max_possible_term)

    kind, value = max_possible_term
    if kind not in ("quantile", "factor"):
        raise ValueError(
            "When it is a tuple, the first value of `max_possible_term` must be "
            "'quantile' or 'factor'."
        )
    off_diag = instance.matrix.numpy()[np.triu_indices(instance.size, k=1)]
    positive = off_diag[off_diag > 0]
    if not positive.size:
        raise ValueError(
            f"Cannot resolve a '{kind}' `max_possible_term` for an instance with no strictly "
            "positive off-diagonal coefficient: set `max_possible_term` to a float instead."
        )
    if kind == "factor":
        return _check_positive_max_possible_term(instance._max_off_diag * value)

    return _check_positive_max_possible_term(float(np.quantile(positive, value)))


def _check_positive_max_possible_term(max_possible_term: float) -> float:
    if max_possible_term <= 0:
        raise ValueError(
            f"`max_possible_term` must resolve to a strictly positive value, got "
            f"{max_possible_term}."
        )
    return max_possible_term


def _number_of_traps_from_device(device: qoolqit.Device) -> int:
    """Determine the number of traps to use based on the device constraints.

    Inspects the device's layout and atom number limits to derive an
    appropriate trap count. The resolution order is:

    1. ``max_layout_traps`` - if the device exposes a hard trap limit, use it directly.
    2. ``max_atom_num`` / ``max_layout_filling`` - if only an atom-number limit is
        available, derive the minimum number of traps needed to accommodate that
        many atoms at the device's maximum filling ratio.
    3. Fallback - return ``200`` when neither property is set.

    Args:
        device (Device): The quantum device whose constraints are inspected.

    Returns:
        int: The number of traps to allocate for the embedding.
    """
    if device._device.max_layout_traps:
        return device._device.max_layout_traps

    if device._device.max_atom_num:
        return int(np.ceil(device._device.max_atom_num / device._device.max_layout_filling))

    return 200


def embed_for_device(
    instance: Instance,
    device: qoolqit.Device,
) -> qoolqit.Register:
    """Embed a QUBO instance using the greedy layout-based algorithm, sized for *device*.

    Convenience wrapper around `embed` that derives `Config.traps` and
    `Config.max_min_dist_ratio` from *device* via [`Config.from_device`][].

    Args:
        instance: The QUBO instance to embed.
        device: Target quantum device the resulting register must fit.

    Returns:
        A register mapping each atom to a 2-D position.

    To also tune device-independent parameters (e.g. `lattice` or
    `max_possible_term`), combine [`Config.from_device`][] with [`embed`][]
    directly:

    Example:
        ```python
        config = Config.from_device(device)
        config.lattice = Lattice.SQUARE
        register = embed(instance, config=config)
        ```
    """
    logger.debug("embed_for_device: instance size=%d, device=%r", instance.size, device)
    return embed(instance, config=Config.from_device(device))


def embed(
    instance: Instance,
    *,
    config: Config | None = None,
) -> qoolqit.Register:
    """Embed a QUBO instance using the greedy layout-based algorithm.

    The algorithm operates entirely in adimensional units (interactions
    scale as ``1 / distance ** 6``), so the coordinates it returns are already
    final and require no post-hoc rescaling.

    Warning:
        A poorly chosen `config` can produce a register that is incompatible
        with a target device. See [`Config`][].

    Args:
        instance: The QUBO instance to embed.  Its ``matrix`` attribute drives
            the greedy cost function.
        config: Greedy embedding parameters, fully resolved (see [`Config.from_device`][]
            for deriving `traps` and `max_min_dist_ratio` from a device).
            ``max_min_dist_ratio`` bounds the ratio between the largest and
            the smallest inter-atom distance in the resulting register.

    Returns:
        A register mapping each atom to a 2-D position.

    Raises:
        ValueError: If `instance` has no variables (``size == 0``), since a
            register must contain at least one qubit. If the resolved trap
            count is less than ``instance.size`` (i.e. there are not enough
            trap sites for all QUBO variables).
    """
    config = config or Config()
    logger.debug("embed: instance size=%d, config=%r", instance.size, config)
    if not instance:
        raise ValueError("Cannot embed an empty instance (size=0): nothing to place.")

    if _has_negative_offdiagonal(instance.matrix):
        raise ValueError("QUBOs with negative off-diagonal coefficients cannot be embedded.")

    if config.traps < instance.size:
        raise ValueError(
            "Number of traps must be at least equal to the number of atoms on the register."
        )

    if instance.size == 1:
        # A single atom has no off-diagonal term to place it relative to,
        # so it is placed at the origin without running the algorithm.
        return qoolqit.Register.from_coordinates(tensor.zeros(1, 2))

    # spacing between adjacent trap sites, derived from the largest QUBO term
    # so that it is exactly representable at the minimum trap-trap distance
    # (interactions scale as 1 / distance ** 6).
    max_possible_term = _resolve_max_possible_term(config.max_possible_term, instance)
    spacing = max_possible_term ** (-1 / 6)

    # build params for the Greedy algorithm
    params = {
        "layout": config.lattice,
        "traps": config.traps,
        "spacing": spacing,
        "norm": config.norm,
        # animation controls (all read by Greedy)
        "draw_steps": config._draw_steps,  # collect per-step data
        "animation": config._draw_steps,  # render animation after run
        "animation_save_path": config._animation_save_path,  # optional export
    }

    # --- Call Greedy (unchanged public signature)
    _, coords = greedy.Greedy().launch_greedy(
        Q=instance.matrix,
        max_min_dist_ratio=config.max_min_dist_ratio,
        params=params,
    )
    register = qoolqit.Register.from_coordinates(coords)
    return register
