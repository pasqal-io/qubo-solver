from __future__ import annotations

from typing import Sequence

import qoolqit
from qoolqit import Constant as ConstantWaveform


def constant_weighted_dmm(
    norm_weights: Sequence[float],
    duration: float,
    final_detuning: float,
) -> qoolqit.drive.DetuningMapModulator:
    """Create a DetuningMapModulator (DMM) object with a single constant waveform, weighted with per-qubit normalized weights (i.e. in [0, 1]).

    The convention required by the qoolqit/Pulser stack is that DMM waveform
    values must be ≤ 0 (i.e., ``final_detuning`` should be negative).

    Args:
        norm_weights (Sequence[float]): Per-qubit normalized weights for the
            DMM, each value in [0, 1].
        duration (float): Waveform duration.
        final_detuning (float): Detuning final value (should be ≤ 0).

    Returns:
        DetuningMapModulator: DetuningMapModulator with a constant
            waveform for QUBO solving.
    """
    waveform = ConstantWaveform(duration, final_detuning)
    return qoolqit.drive.DetuningMapModulator(
        weights={str(i): w for i, w in enumerate(norm_weights)},
        waveform=waveform,
    )
