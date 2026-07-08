"""Greedy embedding algorithm adapter for QUBO instances.

This module wraps the internal greedy embedding and exposes a single `embed`
entry point that accepts a [`qubosolver.Instance`][] and returns a
`qoolqit.Register` ready for use in a quantum program.

The greedy algorithm places logical QUBO nodes one at a time onto trap sites
of a pre-defined lattice (triangular or square), choosing at each step the
(node, trap) pair that minimises the incremental mismatch between the QUBO
coefficient matrix and the physical interaction matrix (∝ C/‖rᵢ − rⱼ‖⁶).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pathlib
import torch

import qoolqit

from ._algorithms import greedy
from qubosolver import Instance, LayoutType, EmbeddingConfig


@dataclass
class Config:
    """Configuration for the greedy embedding algorithm.

    Attributes:
        traps: Number of trap sites in the layout. ``-1`` means auto-detect
            from the device.
        spacing: Minimum distance between adjacent trap sites (µm).
        layout: Lattice layout type (square or triangular).
        draw_steps: If ``True``, collect per-step data for animation.
        animation_save_path: Optional path to save the embedding animation.
    """

    traps: int = -1
    spacing: float = 7.0
    layout: LayoutType = LayoutType.TRIANGULAR
    draw_steps: bool = False
    animation_save_path: pathlib.Path | None = None

    def update_from_device(self, device: qoolqit.Device) -> None:
        """Update trap count and spacing in-place from device constraints.

        When ``traps`` is ``-1`` (auto), resolves the trap count via
        `_number_of_traps_from_device`.  Also raises ``spacing`` to the
        device's ``min_atom_distance`` when that property is available, so the
        resulting layout always satisfies hardware constraints.

        Args:
            device: Target quantum device whose ``_device`` attributes are
                inspected for ``max_layout_traps``, ``max_atom_num``,
                ``max_layout_filling``, and ``min_atom_distance``.
        """
        if self.traps == -1:
            self.traps = _number_of_traps_from_device(device)

        _device = device._device
        if hasattr(_device, "min_atom_distance"):
            self.spacing = max(self.spacing, float(_device.min_atom_distance))

    @staticmethod
    def from_embedding_config(config: EmbeddingConfig) -> Config:
        """Create a [`Config`][] from a user-facing [`EmbeddingConfig`][].

        Maps the ``greedy_*`` fields of *config* onto the corresponding
        `Config` attributes.

        Args:
            config: The embedding configuration to convert.

        Returns:
            A configuration fully populated from the ``greedy_*`` embedding settings of *config*.
        """
        cfg = Config()
        cfg.traps = config.greedy_traps
        cfg.spacing = config.greedy_spacing
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
    normalize: bool = True,
) -> qoolqit.Register:
    """Embed a QUBO instance using the greedy algorithm.

    Runs the greedy placer on the QUBO coefficient matrix.
    Atom labels in the returned register are stringified integer indices
    (``"0"``, ``"1"``, …) matching the variable ordering of the QUBO matrix.

    Two coordinate-scaling modes are supported, selected via *normalize*:

    * **normalize=True** (default) — rescales coordinates so that the
      minimum inter-atom distance is exactly ``1.0001``, the smallest
      separation accepted by normalized Pasqal devices.  Use this when
      ``EmbeddingConfig.min_distance`` is set (heuristic drive-shaping).
    * **normalize=False** — converts the raw greedy coordinates from μm to
      qoolqit's internal distance unit using ``device.converter.factors[2]``.
      Use this when the caller (e.g. the optimised drive-shaping path) will
      handle normalisation externally or does not require a fixed minimum
      distance.

    Args:
        instance: The QUBO instance to embed.  Its ``matrix`` attribute drives
            the greedy cost function.
        device: Target quantum device.
        config: Greedy embedding parameters.  ``update_from_device`` is called
            on this object before the algorithm runs, so device constraints
            are always respected.
        normalize: Controls coordinate post-processing; see above.

    Returns:
        A register mapping each atom label to its 2-D position, with positions determined by the greedy placer.

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
        "device": device._device,
        "layout": config.layout,
        "traps": config.traps,
        "spacing": config.spacing,
        # animation controls (all read by Greedy)
        "draw_steps": config.draw_steps,  # collect per-step data
        "animation": config.draw_steps,  # render animation after run
        "animation_save_path": config.animation_save_path,  # optional export
    }

    # --- Call Greedy (unchanged public signature)
    _, _, coords, _, _ = greedy.Greedy().launch_greedy(
        Q=instance.matrix,
        params=params,
    )
    if normalize:
        min_reg_distance = torch.cdist(coords, coords).fill_diagonal_(float("inf")).min()
        coords *= 1.0001 / min_reg_distance
    else:
        distance_conversion = device.converter.factors[2]
        coords /= distance_conversion

    # build the register (unchanged)
    qubits = {str(i): coord for i, coord in enumerate(coords)}
    register = qoolqit.Register(qubits)
    return register
