from __future__ import annotations

from typing import TypeAlias

from qubosolver import QUBOInstance
from qubosolver.types.label import Labelling, _to_callable
from qoolqit import Register
from qoolqit.embedding import Blade, BladeConfig

#  TODO: Using `type` statement when Python >= 3.12
Config: TypeAlias = BladeConfig


def embed(
    instance: QUBOInstance,
    *,
    config: Config = Config(),
    normalize: bool = True,
    labelling: Labelling = str,
) -> Register:
    """Embed a QUBO instance using the BLaDE algorithm.

    Args:
        instance: The QUBO instance whose coefficient matrix is embedded.
        config: BLaDE configuration (step count, dimensions, etc.).
        normalize: If ``True``, rescale coordinates so that the minimum
            inter-atom spacing is approximately 1.

    Returns:
        A :class:`~qoolqit.Register` with atom positions determined by BLaDE.
    """
    _blade = Blade(config)
    graph = _blade.embed(instance.matrix.numpy())
    if normalize:
        graph.rescale_coords(spacing=1.0001)

    labelling = _to_callable(labelling)
    register = Register({labelling(i): coord for (i, coord) in enumerate(graph.coords.values())})

    return register
