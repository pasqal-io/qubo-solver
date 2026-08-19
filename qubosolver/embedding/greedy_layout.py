"""Greedy embedding algorithm adapter for QUBO instances.

This module wraps the internal greedy embedding and exposes a single `embed`
entry point that accepts a [`qubosolver.Instance`][] and returns a
`qoolqit.Register` ready for use in a quantum program.

The greedy algorithm places logical QUBO nodes one at a time onto trap sites
of a pre-defined lattice (triangular or square), choosing at each step the
(node, trap) pair that minimises the incremental mismatch between the QUBO
coefficient matrix and the physical interaction matrix (∝ 1/‖rᵢ − rⱼ‖⁶).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pathlib

import qoolqit

from ._algorithms import greedy
from qubosolver import Instance, tensor
from .enums import Layout
from .config import Config as EmbeddingConfig
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal


@dataclass
class Config:
    """Configuration for the greedy embedding algorithm.

    Attributes:
        traps: Number of trap sites in the layout. ``"device"`` means
            auto-detect from the device.
        max_possible_term: Largest QUBO interaction term representable at the
            minimum trap-trap distance, in adimensional units. If a float, it
            is used directly. If a tuple, the first element must be
            ``'factor'`` and the second element is a multiplier on the QUBO
            instance's largest off-diagonal coefficient. The corresponding
            spacing is ``max_possible_term ** (-1 / 6)``, since interactions
            scale as ``1 / distance ** 6``.
        layout: Lattice layout type (square or triangular).
        draw_steps: If ``True``, collect per-step data for animation.
        animation_save_path: Optional path to save the embedding animation.
        max_min_dist_ratio: Maximum allowed ratio between the largest and
            the smallest inter-atom distance in the resulting register.
            ``"device"`` means it is derived from the device.
    """

    traps: int | Literal["device"] = "device"
    max_possible_term: float | tuple[Literal["factor"], float] = ("factor", 1.0)
    layout: Layout = Layout.TRIANGULAR
    draw_steps: bool = False
    animation_save_path: pathlib.Path | None = None
    max_min_dist_ratio: float | Literal["device"] = "device"

    def _update_from_device(self, device: qoolqit.Device) -> None:
        """Resolve the ``"device"`` sentinels in-place from device constraints.

        When ``traps`` is ``"device"`` (auto), resolves it via
        `_number_of_traps_from_device`. When ``max_min_dist_ratio`` is
        ``"device"`` (auto), resolves it from *device*'s
        ``max_radial_distance`` / ``min_distance`` specs (or ``inf`` when the
        device imposes no such limits).

        Args:
            device: Target quantum device whose ``_device`` attributes are
                inspected for ``max_layout_traps``, ``max_atom_num``,
                ``max_layout_filling``, ``max_radial_distance``, and
                ``min_distance``.
        """
        if self.traps == "device":
            self.traps = _number_of_traps_from_device(device)

        if self.max_min_dist_ratio == "device":
            specs = device.specs
            min_distance = specs["min_distance"]
            max_radial_distance = specs["max_radial_distance"]
            if min_distance is not None and min_distance > 0 and max_radial_distance is not None:
                self.max_min_dist_ratio = max_radial_distance / min_distance
            else:
                self.max_min_dist_ratio = float("inf")

    @staticmethod
    def _from_embedding_config(config: EmbeddingConfig) -> Config:
        """Create a [`Config`][] from a user-facing [`EmbeddingConfig`][].

        Maps the ``greedy_*`` fields of *config* onto the corresponding
        `Config` attributes. Sentinel values (``"device"`` for ``greedy_traps``,
        ``"device"`` for ``max_min_dist_ratio``) are carried through as
        ``"device"`` and only resolved later, by `update_from_device`.

        Args:
            config: The embedding configuration to convert.

        Returns:
            A configuration fully populated from the ``greedy_*`` embedding settings of *config*.
        """
        cfg = Config()
        cfg.traps = config.greedy_traps
        cfg.max_possible_term = config.greedy_max_possible_term

        cfg.layout = EmbeddingConfig._normalize_layout(config.greedy_layout)
        cfg.draw_steps = config.draw_steps
        path = config.animation_save_path
        cfg.animation_save_path = pathlib.Path(path) if path else None
        cfg.max_min_dist_ratio = config.max_min_dist_ratio

        return cfg


def _resolve_max_possible_term(
    max_possible_term: float | tuple[Literal["factor"], float], instance: Instance
) -> float:
    """Resolve a `Config.max_possible_term` value to a plain float.

    Args:
        max_possible_term: If a float, returned as-is. If a tuple, the first
            element must be ``'factor'`` and the second element is a
            multiplier on *instance*'s largest off-diagonal coefficient.
        instance: The QUBO instance being embedded, used to resolve the
            ``'factor'`` tuple form.

    Returns:
        The resolved maximum representable quadratic term, as a float.

    Raises:
        ValueError: If *max_possible_term* is a tuple whose first element is
            not ``'factor'``, or if *instance* has fewer than 2 variables and
            therefore no off-diagonal coefficient to scale by.
    """
    if isinstance(max_possible_term, float):
        return max_possible_term

    kind, factor = max_possible_term
    if kind != "factor":
        raise ValueError(
            "When it is a tuple, the first value of `max_possible_term` must be 'factor'."
        )
    if instance.size < 2:
        raise ValueError(
            "Cannot resolve a 'factor' `max_possible_term` for an instance with fewer than "
            f"2 variables (size={instance.size}): it has no off-diagonal coefficient to scale."
        )
    return instance._max_off_diag * factor


def _number_of_traps_from_device(device: qoolqit.Device) -> int:
    """Determine the number of traps to use based on the device constraints.

    Inspects the device's layout and atom number limits to derive an
    appropriate trap count. The resolution order is:

    1. ``max_layout_traps`` – if the device exposes a hard trap limit, use it directly.
    2. ``max_atom_num`` / ``max_layout_filling`` – if only an atom-number limit is
        available, derive the minimum number of traps needed to accommodate that
        many atoms at the device's maximum filling ratio.
    3. Fallback – return ``200`` when neither property is set.

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


def embed(
    instance: Instance,
    *,
    device: qoolqit.Device,
    config: Config = Config(),
) -> qoolqit.Register:
    """Embed a QUBO instance using the greedy algorithm.

    The greedy algorithm operates entirely in adimensional units (interactions
    scale as ``1 / distance ** 6``), so the coordinates it returns are already
    final and require no post-hoc rescaling.

    Args:
        instance: The QUBO instance to embed.  Its ``matrix`` attribute drives
            the greedy cost function.
        device: Target quantum device.
        config: Greedy embedding parameters.  ``update_from_device`` is called
            on this object before the algorithm runs, so device constraints
            are always respected. ``max_min_dist_ratio`` bounds the ratio
            between the largest and the smallest inter-atom distance in the
            resulting register.

    Returns:
        A register mapping each atom to a 2-D position.

    Raises:
        ValueError: If *instance* has no variables (``size == 0``), since a
            register must contain at least one qubit. If the resolved trap
            count is less than ``instance.size`` (i.e. there are not enough
            trap sites for all QUBO variables).
    """
    if not instance:
        raise ValueError("Cannot embed an empty instance (size=0): nothing to place.")

    if _has_negative_offdiagonal(instance.matrix):
        raise ValueError("QUBOs with negative off-diagonal coefficients cannot be embedded.")

    config._update_from_device(device)
    assert isinstance(config.traps, int)  # nosec B101
    assert isinstance(config.max_min_dist_ratio, float)  # nosec B101

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
    spacing = max_possible_term ** (-1 / 6) if max_possible_term != 0 else 1

    # build params for the Greedy algorithm
    params = {
        "layout": config.layout,
        "traps": config.traps,
        "spacing": spacing,
        # animation controls (all read by Greedy)
        "draw_steps": config.draw_steps,  # collect per-step data
        "animation": config.draw_steps,  # render animation after run
        "animation_save_path": config.animation_save_path,  # optional export
    }

    # --- Call Greedy (unchanged public signature)
    _, coords = greedy.Greedy().launch_greedy(
        Q=instance.matrix,
        max_min_dist_ratio=config.max_min_dist_ratio,
        params=params,
    )

    # build the register (unchanged)
    qubits = {str(i): coord for i, coord in enumerate(coords)}
    register = qoolqit.Register(qubits)
    return register
