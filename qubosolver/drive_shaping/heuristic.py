from __future__ import annotations

import numpy as np
import torch
import warnings

import qoolqit
from qubosolver import QUBOInstance, Labelling

from ._device_specs import pulser_specs as _pulser_specs
from ._waveforms import constant_weighted_dmm


def build_drive(
    Q: QUBOInstance,
    device: qoolqit.Device,
    *,
    dmm: bool = False,
    kappa: float = 0.25,
    labelling: Labelling = str,
) -> qoolqit.Drive:
    """Generate a heuristic drive schedule for QUBO solving.

    Constructs amplitude and detuning waveforms from the QUBO diagonal
    coefficients.  When DMM is available, per-atom detuning weights are
    computed so that the final local detuning encodes the QUBO diagonal.

    Args:
        register: The physical atom register.
        Q: The QUBO instance whose diagonal encodes target detunings.
        device: Target quantum device (provides hardware limits).
        dmm: Whether to use the Detuning Map Modulator for local control.
        kappa: Ratio between peak Rabi frequency and peak detuning.
            Defaults to 0.25.

    Returns:
        A :class:`~qoolqit.Drive` ready for compilation and execution.
    """
    # Hardware bounds
    specs = device.specs
    max_seq_duration: float = specs["max_duration"] or 1000.0
    pulser_specs = _pulser_specs(device)
    use_dmm = dmm and (pulser_specs["dmm_bottom_detuning"] is not None)

    max_amplitude = specs["max_amplitude"] or 1.0
    max_abs_detuning = specs["max_abs_detuning"] or 9.0

    det_amp_ratio = max_amplitude / max_abs_detuning
    if kappa < det_amp_ratio:
        warnings.warn(
            f"heuristic_kappa is too small ({kappa}), you're likely to get a qoolqit CompilationError. Set it above {det_amp_ratio}."
        )

    n = Q.size

    # Target local final detunings
    d = (-0.5 * torch.diag(Q._normalized_matrix)).cpu().numpy()
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

    omega_max = kappa * np.max(np.abs(d))
    # How to get max detuning ?
    delta_0 = -np.max(np.abs(d))

    # Amplitude waveform: 0 -> plateau -> 0
    eps = 1e-9
    amp_wave = qoolqit.Interpolated(
        max_seq_duration,
        [eps, omega_max, omega_max, eps],
    )

    # Global detuning waveform: initial negative -> final target
    det_wave = qoolqit.Interpolated(
        max_seq_duration,
        [delta_0, delta_0, delta_g_T, delta_g_T],
    )

    # DMM weighted detunings
    wdetunings = None
    if use_dmm:
        wdetunings = constant_weighted_dmm(
            weights,
            max_seq_duration,
            final_detuning=delta_dmm_T,
            labelling=labelling,
        )

    return qoolqit.Drive(
        amplitude=amp_wave,
        detuning=det_wave,
        dmm=wdetunings,
    )
