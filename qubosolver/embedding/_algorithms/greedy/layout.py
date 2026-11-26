# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Compact trap-layout generation for the greedy embedding algorithm."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from pulser.register.special_layouts import SquareLatticeLayout, TriangularLatticeLayout

from qubosolver.embedding.enums import Lattice

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_layout(
    *, layout_type: Lattice | str = Lattice.TRIANGULAR, n_traps: int
) -> NDArray[np.float64]:
    """Build a lattice of `n_traps` unit-spacing trap coordinates.

    For a square lattice, builds a grid large enough to contain `n_traps`
    points on a disk inscribed in the square (`n = ceil(sqrt(2 * n_traps))`)
    and keeps only the `n_traps` points closest to the origin, so the
    resulting layout is compact (disk-like, centered) rather than an
    arbitrary corner block of the grid.

    Args:
        layout_type: Lattice type, `Lattice.TRIANGULAR`/`Lattice.SQUARE`
            or their lowercase string names. Defaults to `Lattice.TRIANGULAR`.
        n_traps: Number of trap sites to return.

    Returns:
        An array of shape `(n_traps, 2)` with unit-spacing trap coordinates.

    Raises:
        ValueError: If `layout_type` is not a recognized layout.
    """
    if isinstance(layout_type, str):
        layout_type = layout_type.lower()

    if layout_type not in [
        Lattice.TRIANGULAR,
        "triangular",
        Lattice.SQUARE,
        "square",
    ]:
        msg = f"Unsupported layout_type: {layout_type!r}"
        raise ValueError(msg)

    if n_traps == 0:
        return np.zeros((0, 2))

    match layout_type:
        case Lattice.TRIANGULAR | "triangular":
            return np.asarray(TriangularLatticeLayout(n_traps, spacing=1).coords, dtype=np.float64)

        case Lattice.SQUARE | "square":
            n = math.ceil(math.sqrt(2 * n_traps))
            coords = np.asarray(SquareLatticeLayout(n, n, spacing=1).coords, dtype=np.float64)
            squared_distances = np.square(coords).sum(axis=1)
            return coords[np.argsort(squared_distances, kind="stable")[:n_traps]]

        case _:  # no cov
            # Unreachable: layout_type was already validated above.
            msg = f"Unhandled layout_type: {layout_type!r}"
            raise AssertionError(msg)
