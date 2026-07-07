"""BLaDE (Balanced Layout and Distance Embedding) adapter for QUBO instances.

This module is a thin wrapper around :class:`qoolqit.embedding.Blade` that
exposes a single `embed` entry point accepting a
:class:`~qubosolver.types.Instance` and returning a
:class:`~qoolqit.Register` ready for use in a quantum program.

BLaDE maps the QUBO coefficient matrix onto a 2-D (or higher-dimensional)
set of atom positions so that the physical interaction strengths
(∝ 1/‖rᵢ − rⱼ‖⁶) are as proportional to the QUBO edge weights as possible.
It does so by iteratively refining coordinates across multiple dimensional
reduction rounds.

Typical usage goes through :class:`~qubosolver.embedding.BLaDEmbedder`, which
reads :class:`~qubosolver.config.EmbeddingConfig` parameters and calls
`embed` directly.
"""

from __future__ import annotations

from typing import TypeAlias

from qubosolver import Instance
from qoolqit import Register
from qoolqit.embedding import Blade, BladeConfig

# Alias BladeConfig under the module-local name ``Config`` so callers can
# refer to ``blade.Config`` without importing from qoolqit directly.
# TODO: Replace TypeAlias with the ``type`` statement when Python >= 3.12.
Config: TypeAlias = BladeConfig


def embed(
    instance: Instance,
    *,
    config: Config = Config(),
    normalize: bool = True,
) -> Register:
    """Embed a QUBO instance using the BLaDE algorithm.

    Runs the BLaDE optimisation on the QUBO coefficient matrix and converts
    the resulting graph coordinates into a :class:`~qoolqit.Register`.  Atom
    labels are assigned as stringified integer indices (``"0"``, ``"1"``, …)
    matching the variable ordering of the QUBO matrix.

    Args:
        instance: The QUBO instance to embed.  Its ``matrix`` attribute is
            passed directly to the BLaDE optimiser as a NumPy array.
        config: BLaDE configuration controlling the optimisation (number of
            steps per round, initial atom positions, dimension sequence,
            maximum allowed ratio of radial to minimum distance, etc.).
            Defaults to :class:`Config` with its own defaults.
        normalize: If ``True``, rescale the final atom coordinates so that the
            minimum inter-atom distance is exactly ``1.0001`` — the
            smallest separation accepted by normalized Pasqal devices.

    Returns:
        A :class:`~qoolqit.Register` mapping each atom label to its 2-D
        position, with atom positions determined by BLaDE.
    """
    _blade = Blade(config)
    graph = _blade.embed(instance.matrix.numpy())
    if normalize:
        graph.rescale_coords(spacing=1.0001)

    register = Register({str(i): coord for (i, coord) in enumerate(graph.coords.values())})

    return register
