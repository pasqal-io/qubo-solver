from enum import Enum
import torch

from pulser.register.special_layouts import SquareLatticeLayout, TriangularLatticeLayout
from pulser.register.register_layout import RegisterLayout


class LayoutType(Enum):
    """
    Type of layout for the greedy embedding method
    """

    SQUARE = SquareLatticeLayout
    TRIANGULAR = TriangularLatticeLayout


def get_layout(*, layout_type: LayoutType = LayoutType.TRIANGULAR, n_traps: int) -> torch.Tensor:
    """Return `n_traps` 2D trap coordinates on a specified lattice with unit spacing.

    Parameters:
        layout_type:
            Layout family to use. Accepts a `LayoutType` or the strings "triangular" / "square"
            (case-insensitive).
        n_traps:
            Number of traps (points) to return. Must be a strictly positive integer.

    Returns:
        torch.Tensor
            Tensor of shape (n_traps, 2) containing (x, y) coordinates, with lattice spacing 1.

    Notes:
        - Triangular: uses Pulser's TriangularLatticeLayout(n_traps, spacing=1).
        - Square: builds an n x n square lattice with spacing 1 (n = ceil(sqrt(2*n_traps))) then
        selects the n_traps points with smallest squared distance to the origin (compact set).
    """

    if isinstance(layout_type, str):
        layout_type = layout_type.lower()

    layout: RegisterLayout

    match layout_type:
        case LayoutType.TRIANGULAR | "triangular":
            layout = LayoutType.TRIANGULAR.value(n_traps, spacing=1)
            return torch.tensor(layout.coords)

        case LayoutType.SQUARE | "square":
            n = int(torch.ceil(torch.sqrt(2 * torch.tensor(n_traps))).item())
            layout = LayoutType.SQUARE.value(n, n, spacing=1)
            coords = torch.tensor(layout.coords)
            squared_distances = (coords**2).sum(dim=1)
            return coords[torch.argsort(squared_distances)[:n_traps]]
