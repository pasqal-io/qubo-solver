from __future__ import annotations

import warnings

import numpy as np
from qoolqit.devices import Device
from qoolqit.register import Register
from qoolqit.drive import DetuningMapModulator
from qoolqit import Constant as ConstantWaveform


def _clip_final_detuning_to_dmm_budget(
    device: Device | None,
    norm_weights: list[float],
    final_detuning: float,
    energy_scale: float | None,
) -> float:
    """Clip a DMM's final detuning so its compiled version
    respects the device's ``total_bottom_detuning`` budget.

    Args:
        device (Device | None): The target device. If ``None``, no clipping
            is applied.
        norm_weights (list[float]): Per-qubit normalized DMM weights.
        final_detuning (float): The unclipped DMM final detuning value.
        energy_scale (float | None): The predicted ``MAX_ENERGY`` rescale
            factor (``device._target_amp / drive.amplitude.max()``). If
            ``None``, no clipping is applied.

    Returns:
        float: ``final_detuning``, clipped if it would otherwise violate the
            device's ``total_bottom_detuning`` budget.
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
            f"{-max_abs_final_detuning:.3f}. Local detuning precision is reduced "
            "for this instance."
        )
        return -max_abs_final_detuning

    return final_detuning


def constant_weighted_dmm(
    embedding: Register,
    duration: float,
    norm_weights: list[float],
    final_detuning: float,
    device: Device | None = None,
    energy_scale: float | None = None,
) -> DetuningMapModulator | None:
    """Create a DetuningMapModulator (DMM) object with a single constant waveform, weighted with per-qubit normalized weights (i.e. in [0, 1]).

    The convention required by the qoolqit/Pulser stack is that DMM waveform
    values must be ≤ 0 (i.e., ``final_detuning`` should be negative).

    Args:
        embedding (Register): embedding targeted.
        duration (float): Waveform duration.
        norm_weights (list[float]): Normalized weights for DMM.
        final_detuning (float): Detuning final value.
        device (Device | None, optional): target device. When provided
            together with ``energy_scale``, ``final_detuning`` is clipped so
            the compiled sequence respects the device's
            ``total_bottom_detuning`` DMM budget.
            Defaults to ``None`` (no clipping).
        energy_scale (float | None, optional): predicted ``MAX_ENERGY``
            compiler rescale factor (``device._target_amp /
            drive.amplitude.max()``). Required together with ``device`` to
            enable clipping. Defaults to ``None``.

    Returns:
        DetuningMapModulator: DetuningMapModulator with a constant
            waveform for QUBO solving if some dmm weights are non zero,
            None otherwise.
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
    return DetuningMapModulator(
        weights={embedding.qubits_ids[i]: w for i, w in enumerate(norm_weights)},
        waveform=waveform,
    )
