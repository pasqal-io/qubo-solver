"""Compact trap-layout generation for the greedy embedding algorithm."""

from __future__ import annotations

import torch

from qubosolver.types.enums import LayoutType


def get_layout(*, layout_type: LayoutType | str = LayoutType.TRIANGULAR, n_traps: int) -> torch.Tensor:
    """Build a lattice of `n_traps` unit-spacing trap coordinates.

    For a square lattice, builds a grid large enough to contain `n_traps`
    points on a disk inscribed in the square (`n = ceil(sqrt(2 * n_traps))`)
    and keeps only the `n_traps` points closest to the origin, so the
    resulting layout is compact (disk-like, centered) rather than an
    arbitrary corner block of the grid.

    Args:
        layout_type: Lattice type, `LayoutType.TRIANGULAR`/`LayoutType.SQUARE`
            or their lowercase string names. Defaults to `LayoutType.TRIANGULAR`.
        n_traps: Number of trap sites to return.

    Returns:
        A tensor of shape `(n_traps, 2)` with unit-spacing trap coordinates.

    Raises:
        ValueError: If `layout_type` is not a recognized layout.
    """
    if isinstance(layout_type, str):
        layout_type = layout_type.lower()

    match layout_type:
        case LayoutType.TRIANGULAR | "triangular":
            layout = LayoutType.TRIANGULAR.value(n_traps, spacing=1)
            return torch.tensor(layout.coords)

        case LayoutType.SQUARE | "square":
            n = int(torch.ceil(torch.sqrt(2 * torch.tensor(n_traps))).item())
            layout = LayoutType.SQUARE.value(n, n, spacing=1)
            coords = torch.tensor(layout.coords)
            squared_distances = coords.square().sum(dim=1)
            return coords[torch.argsort(squared_distances)[:n_traps]]

        case _:
            raise ValueError(f"Unsupported layout_type: {layout_type!r}")
