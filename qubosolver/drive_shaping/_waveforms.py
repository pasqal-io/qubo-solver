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
        norm_weights (Sequence[float]): Per-qubit normalized weights for the
            DMM, each value in [0, 1].
        duration (float): Waveform duration.
        final_detuning (float): Detuning final value (should be ≤ 0).
        labelling (Labelling): Callable used to map qubit indices to qubit
            labels. Defaults to ``str``. Can be inferred from a register via
            ``register.qubits_ids``.

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
