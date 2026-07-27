from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import qoolqit
from qoolqit import Constant as ConstantWaveform


def _clip_final_detuning_to_dmm_budget(
    device: qoolqit.Device | None,
    norm_weights: Sequence[float],
    final_detuning: float,
    energy_scale: float | None,
) -> float:
    """Clip `final_detuning` so the compiled DMM stays within the device's DMM budget.

    The `MAX_ENERGY` compiler profile rescales the whole sequence's energies
    by `energy_scale` at compile time, so the physical (pre-scaling)
    `final_detuning` value passed here must be checked against the device's
    `total_bottom_detuning` budget divided by that scale factor. A ``0.999``
    margin guards against rounding pushing the compiled value just over the
    hard device limit.

    Args:
        device: Target quantum device, or `None` to skip clipping.
        norm_weights: Per-qubit normalized DMM weights.
        final_detuning: Requested detuning final value (should be ≤ 0).
        energy_scale: Predicted `MAX_ENERGY` compiler rescale factor, or
            `None` to skip clipping.

    Returns:
        The (possibly clipped) final detuning value.
    """
    if device is None or energy_scale is None:
        return final_detuning

    sum_weight = sum(norm_weights)
    dmm_channels = list(device._device.dmm_channels.values())
    total_bottom_detuning = dmm_channels[0].total_bottom_detuning if dmm_channels else None

    if total_bottom_detuning is None:
        return final_detuning

    safe_total_bottom_detuning = 0.999 * total_bottom_detuning  # margin for rounding
    max_abs_final_detuning = abs(safe_total_bottom_detuning) / (sum_weight * energy_scale)

    if abs(final_detuning) > max_abs_final_detuning:
        warnings.warn(
            "DMM final detuning would exceed the device's total_bottom_detuning "
            f"budget ({total_bottom_detuning} rad/us) once compiled "
            f"(sum(weights)={sum_weight:.3f}); clipping from {final_detuning:.3f} to "
            f"{-max_abs_final_detuning:.3f}. Local detuning precision is reduced for this instance."
        )
        return -max_abs_final_detuning

    return final_detuning


def constant_weighted_dmm(
    norm_weights: Sequence[float],
    duration: float,
    final_detuning: float,
    device: qoolqit.Device | None = None,
    energy_scale: float | None = None,
) -> qoolqit.drive.DetuningMapModulator | None:
    """Create a DetuningMapModulator (DMM) object with a single constant waveform, weighted with per-qubit normalized weights (i.e. in [0, 1]).

    The convention required by the qoolqit/Pulser stack is that DMM waveform
    values must be ≤ 0 (i.e., ``final_detuning`` should be negative).

    Args:
        norm_weights (Sequence[float]): Per-qubit normalized weights for the
            DMM, each value in [0, 1].
        duration (float): Waveform duration.
        final_detuning (float): Detuning final value (should be ≤ 0).
        device: Target quantum device, used to clip `final_detuning` to the
            device's DMM channel budget. `None` skips clipping.
        energy_scale: Predicted `MAX_ENERGY` compiler rescale factor, used
            together with `device` to clip `final_detuning`. `None` skips
            clipping.

    Returns:
        DetuningMapModulator: DetuningMapModulator with a constant
            waveform for QUBO solving, or `None` when every weight is zero
            (in which case no DMM contributes anything).
    """
    if not np.any(norm_weights):
        return None

    final_detuning = _clip_final_detuning_to_dmm_budget(
        device=device,
        norm_weights=norm_weights,
        final_detuning=final_detuning,
        energy_scale=energy_scale,
    )

    waveform = ConstantWaveform(duration, final_detuning)
    return qoolqit.drive.DetuningMapModulator(
        weights={str(i): w for i, w in enumerate(norm_weights)},
        waveform=waveform,
    )
