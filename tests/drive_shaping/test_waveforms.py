# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import pytest
import pytest_check as check
import qoolqit

from qubosolver.drive_shaping._waveforms import _clip_final_detuning_to_dmm_budget


@pytest.mark.parametrize(
    ("device", "energy_scale"),
    [
        (None, 1.0),
        (qoolqit.AnalogDeviceWithDMM(), None),
        # No DMM channel, hence no detuning budget.
        (qoolqit.AnalogDevice(), 1.0),
    ],
)
def test_clip_final_detuning_skipped(
    device: qoolqit.Device | None, energy_scale: float | None
) -> None:
    final_detuning = _clip_final_detuning_to_dmm_budget(
        device=device, norm_weights=[1.0, 0.5], final_detuning=-1e6, energy_scale=energy_scale
    )
    check.equal(final_detuning, -1e6)
