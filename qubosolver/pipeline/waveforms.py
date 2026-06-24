from __future__ import annotations

from qoolqit.register import Register
from qoolqit.drive import DetuningMapModulator
from qoolqit import Constant as ConstantWaveform


def constant_weighted_dmm(
    embedding: Register,
    duration: float,
    norm_weights: list[float],
    final_detuning: float,
) -> DetuningMapModulator:
    """Create a DetuningMapModulator (DMM) object with a single constant waveform, weighted with per-qubit normalized weights (i.e. in [0, 1]).

    The convention required by the qoolqit/Pulser stack is that DMM waveform
    values must be ≤ 0 (i.e., ``final_detuning`` should be negative).

    Args:
        embedding (Register): embedding targeted.
        duration (float): Waveform duration.
        norm_weights (list[float]): Normalized weights for DMM.
        final_detuning (float): Detuning final value.

    Returns:
        DetuningMapModulator: DetuningMapModulator with a constant
            waveform for QUBO solving.
    """
    waveform = ConstantWaveform(duration, final_detuning)
    return DetuningMapModulator(
        weights={embedding.qubits_ids[i]: w for i, w in enumerate(norm_weights)},
        waveform=waveform,
    )
