from __future__ import annotations

from typing import Sequence

import qoolqit
from qoolqit import Constant as ConstantWaveform

from qubosolver import Labelling
from qubosolver.types.label import _to_callable


def constant_weighted_dmm(
    norm_weights: Sequence[float],
    duration: float,
    final_detuning: float,
    *,
    labelling: Labelling = str,
) -> qoolqit.drive.DetuningMapModulator:
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
    labelling = _to_callable(labelling)
    waveform = ConstantWaveform(duration, final_detuning)
    return qoolqit.drive.DetuningMapModulator(
        weights={labelling(i): w for i, w in enumerate(norm_weights)},
        waveform=waveform,
    )
