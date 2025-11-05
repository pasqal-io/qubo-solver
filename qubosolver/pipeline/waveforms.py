from __future__ import annotations

from qoolqit.register import Register
from qoolqit.drive import WeightedDetuning
from qoolqit.waveforms import Constant


def weighted_detunings(
    embedding: Register,
    duration: float,
    norm_weights: list[float],
    final_detuning: float,
) -> list[WeightedDetuning]:
    """Create the list of weighted detuning for a drive.

    Args:
        embedding (Register): embedding targeted.
        duration (float): Waveform duration.
        norm_weights (list[float]): Normalized weights for DMM.
        final_detuning (float): Detuning final value.

    Returns:
        list[WeightedDetuning]: A list of WeightedDetuning with a constant
            waveform for QUBO solving.
    """
    waveform = Constant(duration, final_detuning)
    return [
        WeightedDetuning(
            weights={embedding.qubits_ids[i]: w for i, w in enumerate(norm_weights)},
            waveform=waveform,
        )
    ]
