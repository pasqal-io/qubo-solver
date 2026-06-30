from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pathlib
import torch

import qoolqit

from ._algorithms import greedy
from qubosolver import QUBOInstance, LayoutType, EmbeddingConfig
from qubosolver.types.label import Labelling, _to_callable


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
        """Update trap count and spacing from device constraints.

        Args:
            device: Target quantum device.
        """
        if self.traps == -1:
            self.traps = _number_of_traps_from_device(device)

        _device = device._device
        if hasattr(_device, "min_atom_distance"):
            self.spacing = max(self.spacing, float(_device.min_atom_distance))

    @staticmethod
    def from_embedding_config(config: EmbeddingConfig) -> Config:
        """Create a :class:`Config` from a user-facing :class:`EmbeddingConfig`.

        Args:
            config: The embedding configuration to convert.

        Returns:
            A :class:`Config` populated from the embedding settings.
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
    instance: QUBOInstance,
    device: qoolqit.Device,
    *,
    config: Config = Config(),
    normalize: bool = True,
    labelling: Labelling = str,
) -> qoolqit.Register:
    """Embed a QUBO instance using the greedy algorithm.

    Places logical nodes one at a time onto trap sites, minimising
    the incremental mismatch between the QUBO coefficient matrix and
    the physical interaction matrix.

    Args:
        instance: The QUBO instance to embed.
        device: Target quantum device (provides layout and distance constraints).
        config: Greedy embedding parameters.
        normalize: If ``True``, rescale coordinates so that the minimum
            inter-atom spacing is approximately 1.

    Returns:
        A :class:`~qoolqit.Register` with atom positions.

    Raises:
        ValueError: If the number of traps is smaller than the problem size.
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

    labelling = _to_callable(labelling)
    # build the register (unchanged)
    qubits = {labelling(i): coord for i, coord in enumerate(coords)}
    register = qoolqit.Register(qubits)
    return register
