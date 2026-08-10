"""Heuristic drive schedule generation for QUBO solving."""

from __future__ import annotations

import logging

import numpy as np
import torch

import qoolqit
from qubosolver import Instance

from ._device_specs import (
    pulser_specs as _pulser_specs,
    max_virtual_amplitude,
    detuning_amplitude_ratio,
)
from ._waveforms import constant_weighted_dmm

logger = logging.getLogger(__name__)


def build_drive(
    instance: Instance,
    register: qoolqit.Register,
    *,
    device: qoolqit.Device,
    dmm: bool = False,
    kappa: float = 0.25,
) -> qoolqit.Drive:
    """Generate a heuristic drive schedule for QUBO solving.

    Constructs amplitude and detuning waveforms from the QUBO diagonal
    coefficients, clamped so they stay within what the compiler can realize
    on `device` for `register`.  When DMM is available, per-atom detuning
    weights are computed so that the final local detuning encodes the QUBO
    diagonal.

    Args:
        instance: The QUBO instance whose diagonal encodes target detunings.
        register: The physical register the drive will run on.
        device: Target quantum device (provides hardware limits).
        dmm: Whether to use the Detuning Map Modulator for local control.
        kappa: Ratio between peak Rabi frequency and peak detuning.

    Returns:
        A drive ready for compilation and execution.
    """
    # Hardware bounds
    specs = device.specs
    max_seq_duration: float = specs["max_duration"] or 1000.0
    pulser_specs = _pulser_specs(device)
    use_dmm = dmm and (pulser_specs["dmm_bottom_detuning"] is not None)

    if specs.get("max_amplitude") is not None and specs.get("max_abs_detuning") is not None:
        max_amplitude = specs["max_amplitude"]
        max_abs_detuning = specs["max_abs_detuning"]
        assert max_amplitude is not None and max_abs_detuning is not None
        det_amp_ratio = max_amplitude / max_abs_detuning
        if kappa < det_amp_ratio:
            logger.warning(
                f"heuristic_kappa is too small ({kappa}), you're likely to get a qoolqit CompilationError. Set it above {det_amp_ratio}."
            )

    n = instance.size

    # Target local final detunings
    d = (-0.5 * torch.diag(instance.matrix)).cpu().numpy()
    d_min = np.min(d)
    d_max = np.max(d)

    omega_max = kappa * np.max(np.abs(d))

    max_amplitude = max_virtual_amplitude(device, register)
    if omega_max > max_amplitude:
        logger.info(
            f"The heuristic drive amplitude ({omega_max}) exceeds the maximum "
            f"amplitude compilable on the device for this register "
            f"({max_amplitude}); clamping to it."
        )
        omega_max = max_amplitude

    max_detuning = detuning_amplitude_ratio(device) * omega_max * (1.0 - 1e-3)
    max_abs_d = float(np.max(np.abs(d)))
    if max_abs_d > max_detuning:
        logger.info(
            f"The heuristic detuning ({max_abs_d}) exceeds the maximum detuning "
            f"compilable on the device for this amplitude ({max_detuning}); "
            f"scaling the detuning down."
        )
        d = d * (max_detuning / max_abs_d)
        d_min = np.min(d)
        d_max = np.max(d)

    if use_dmm:
        # Final global detuning is the top value, DMM pulls down locally
        delta_g_T = d_max

        # DMM convention required by WeightedDetuning:
        # waveform must be <= 0
        spread = max(0.0, d_max - d_min)
        # if spread > 1e-15 and delta_dmm_max > 0.0:
        if spread > 1e-15:
            delta_dmm_T = -spread  # must be <= 0
            denom = d_max - d_min
            weights = ((d_max - d) / denom).clip(0.0, 1.0).tolist()
        else:
            use_dmm = False
            delta_dmm_T = 0.0
            weights = [0.0] * n
    else:
        # No DMM: use a single global final detuning
        delta_g_T = np.mean(d)
        delta_dmm_T = 0.0
        weights = [0.0] * n

    # How to get max detuning ?
    delta_0 = -np.max(np.abs(d))

    # Amplitude waveform: 0 -> plateau -> 0
    eps = 1e-9
    amp_wave = qoolqit.InterpolatedWaveform(
        max_seq_duration,
        [eps, omega_max, omega_max, eps],
    )

    # Global detuning waveform: initial negative -> final target
    det_wave = qoolqit.InterpolatedWaveform(
        max_seq_duration,
        [delta_0, delta_0, delta_g_T, delta_g_T],
    )

    # DMM weighted detunings
    wdetunings = None
    if use_dmm:
        energy_scale = device._target_amp / omega_max
        wdetunings = constant_weighted_dmm(
            weights,
            max_seq_duration,
            final_detuning=delta_dmm_T,
            device=device,
            energy_scale=energy_scale,
        )

    return qoolqit.Drive(
        amplitude=amp_wave,
        detuning=det_wave,
        dmm=wdetunings,
    )
