from __future__ import annotations

import numpy as np
import pytest

import qoolqit
from qubosolver import matrix, tensor, Instance, solving, embedding


def test_too_large_register() -> None:
    # In the quantum pipeline, it's easy to forget to pass the target device to blade's
    # embedding config. The resulting register can then have a too high max/min distance
    # ratio for that device, which without the new exception would only surface as a
    # hard-to-pinpoint compile error deep inside qoolqit.

    np.random.seed(1048)

    instance = Instance(matrix.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))

    # Two close atoms and one far one: a large max/min distance ratio that a
    # device-aware blade config would have avoided.
    coords = tensor.tensor(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, 9.9],
        ]
    )
    config = embedding.blade.Config(
        starting_positions=coords.numpy(),
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
