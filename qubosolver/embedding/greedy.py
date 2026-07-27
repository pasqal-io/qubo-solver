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
import numpy as np
import pathlib

import qoolqit

from ._algorithms import greedy
from qubosolver import Instance, LayoutType, EmbeddingConfig


@dataclass
class Config:
    """Configuration for the greedy embedding algorithm.

    Attributes:
        traps: Number of trap sites in the layout. ``-1`` means auto-detect
            from the device.
        spacing: Minimum distance between adjacent trap sites, in adimensional
            units (derived from the largest representable QUBO term).
        layout: Lattice layout type (square or triangular).
        draw_steps: If ``True``, collect per-step data for animation.
        animation_save_path: Optional path to save the embedding animation.
    """

    traps: int = -1
    spacing: float = 1.0
    layout: LayoutType = LayoutType.TRIANGULAR
    draw_steps: bool = False
    animation_save_path: pathlib.Path | None = None

    def update_from_device(self, device: qoolqit.Device) -> None:
        """Update the trap count in-place from device constraints.

        When ``traps`` is ``-1`` (auto), resolves the trap count via
        `_number_of_traps_from_device`.

        Args:
            device: Target quantum device whose ``_device`` attributes are
                inspected for ``max_layout_traps``, ``max_atom_num``, and
                ``max_layout_filling``.
        """
        if self.traps == -1:
            self.traps = _number_of_traps_from_device(device)

    @staticmethod
    def from_embedding_config(config: EmbeddingConfig, instance: Instance) -> Config:
        """Create a [`Config`][] from a user-facing [`EmbeddingConfig`][].

        Maps the ``greedy_*`` fields of *config* onto the corresponding
        `Config` attributes. ``spacing`` is derived from
        ``greedy_max_possible_term`` so that the largest QUBO interaction term
        is exactly representable at the minimum trap-trap distance
        (``spacing = max_possible_term ** (-1 / 6)``, since interactions scale
        as ``1 / distance ** 6``).

        Args:
            config: The embedding configuration to convert.
            instance: The QUBO instance being embedded, used to resolve
                ``greedy_max_possible_term`` when expressed as a factor of the
                instance's largest off-diagonal coefficient.

        Returns:
            A configuration fully populated from the ``greedy_*`` embedding settings of *config*.
        """
        cfg = Config()
        cfg.traps = config.greedy_traps

        max_possible_term_config = config.greedy_max_possible_term
        if isinstance(max_possible_term_config, float):
            max_possible_term = max_possible_term_config
        else:
            kind, factor = max_possible_term_config
            if kind != "factor":
                raise ValueError(
                    "When it is a tuple, the first value of `greedy_max_possible_term` "
                    "must be 'factor'."
                )
            max_possible_term = instance._max_off_diag * factor
        cfg.spacing = max_possible_term ** (-1 / 6) if max_possible_term != 0 else 1

        cfg.layout = EmbeddingConfig._normalize_layout(config.greedy_layout)
        cfg.draw_steps = config.draw_steps
        path = config.animation_save_path
        cfg.animation_save_path = pathlib.Path(path) if path else None

        return cfg


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
    device: qoolqit.Device,
    *,
    config: Config = Config(),
    max_min_dist_ratio: float,
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
            are always respected.
        max_min_dist_ratio: Maximum allowed ratio between the largest and the
            smallest inter-atom distance in the resulting register.

    Returns:
        A register mapping each atom to a 2-D position.

    Raises:
        ValueError: If the resolved trap count is less than ``instance.size``
            (i.e. there are not enough trap sites for all QUBO variables).
    """
    config.update_from_device(device)

    if config.traps < instance.size:
        raise ValueError(
            "Number of traps must be at least equal to the number of atoms on the register."
        )

    # build params for the Greedy algorithm
    params = {
        "layout": config.layout,
        "traps": config.traps,
        "spacing": config.spacing,
        # animation controls (all read by Greedy)
        "draw_steps": config.draw_steps,  # collect per-step data
        "animation": config.draw_steps,  # render animation after run
        "animation_save_path": config.animation_save_path,  # optional export
    }

    # --- Call Greedy (unchanged public signature)
    _, coords = greedy.Greedy().launch_greedy(
        Q=instance.matrix,
        max_min_dist_ratio=max_min_dist_ratio,
        params=params,
    )

    # build the register (unchanged)
    qubits = {str(i): coord for i, coord in enumerate(coords)}
    register = qoolqit.Register(qubits)
    return register
