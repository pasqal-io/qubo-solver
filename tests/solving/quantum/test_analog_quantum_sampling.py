# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check
import qoolqit

from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    analysis,
    drive_shaping,
    embedding,
    solving,
)
from qubosolver.utils._random import manual_seed


@pytest.mark.duration(3)
def test_too_large_register() -> None:
    # In the quantum pipeline, it's easy to forget to pass the target device to blade's
    # embedding config. The resulting register can then have a too high max/min distance
    # ratio for that device, which without the new exception would only surface as a
    # hard-to-pinpoint compile error deep inside qoolqit.

    np.random.seed(1048)

    instance = Instance(np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64))

    # Two close atoms and one far one: a large max/min distance ratio that a
    # device-aware blade config would have avoided. These are passed as
    # starting_positions rather than left for blade to optimize from scratch,
    # because blade may otherwise settle on a (less optimal but still valid)
    # layout that isn't spread out enough to reproduce the ratio violation.
    coords = np.asarray(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, 9.9],
        ]
    )
    config = embedding.blade.Config(
        starting_positions=coords,
    )

    register = embedding.blade.embed(instance, config=config)

    device = qoolqit.AnalogDevice()
    max_duration = device.specs["max_duration"]
    max_amplitude = device.specs["max_amplitude"]
    assert max_duration is not None
    assert max_amplitude is not None
    drive = qoolqit.Drive(
        amplitude=qoolqit.ConstantWaveform(0.9 * max_duration, 0.1 * max_amplitude)
    )

    with pytest.raises(ValueError, match="max/min distance ratio"):
        solving.analog_quantum_sampling.compile(register, drive, device)


@pytest.mark.usefixtures("restore_rng_state")
def test_single_atom_register() -> None:
    manual_seed(156617)

    instance = Instance(np.asarray([[-2.0]]))

    device = qoolqit.AnalogDeviceWithDMM()
    register = qoolqit.Register.from_coordinates(np.zeros((1, 2)))
    drive = drive_shaping.local_energy_scale.build_drive(instance, register, device=device)
    program = solving.analog_quantum_sampling.compile(register, drive, device)
    job = LocalEmulator().run(program)
    solution = Solution.from_results(job.results(), instance)

    check.equal(solution[0].string, "1")
    check.equal(solution[0].cost, -2.0)
    check.greater(solution[0].probability, 0.7)

    print(analysis.to_dataframe([solution]))


@pytest.mark.parametrize(("x", "fits"), [(3.0, True), (100.0, False)])
def test_single_atom_register_radial_distance(x: float, fits: bool) -> None:
    # A single atom has no minimum distance: its radial distance is measured in units
    # of the device's minimum distance.
    instance = Instance(np.asarray([[-2.0]]))
    device = qoolqit.AnalogDeviceWithDMM()
    register = qoolqit.Register.from_coordinates(np.asarray([[x, 0.0]], dtype=np.float64))
    drive = drive_shaping.local_energy_scale.build_drive(instance, register, device=device)

    if fits:
        solving.analog_quantum_sampling.compile(register, drive, device)
    else:
        with pytest.raises(ValueError, match="max/min distance ratio"):
            solving.analog_quantum_sampling.compile(register, drive, device)
