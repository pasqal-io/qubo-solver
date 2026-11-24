# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import pytest_check as check
import qoolqit
from qoolqit.graphs import DataGraph
from scipy.spatial.distance import pdist, squareform

from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    drive_shaping,
    embedding,
    matrix,
    solving,
)


def test_proportional_diagonal_drive_on_digital_analog_device(
    simple_qubo_instance: Instance,
) -> None:
    register = qoolqit.Register({0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 3.0)})
    drive = drive_shaping.proportional_diagonal.build_drive(
        simple_qubo_instance, register, device=qoolqit.DigitalAnalogDevice(), kappa=0.25
    )

    # DigitalAnalogDevice has a hardcoded duration
    check.almost_equal(drive.duration, 1000.0, abs=1.0e-3)

    check.equal(drive.amplitude.duration, drive.duration)
    check.almost_equal(drive.amplitude.min(), 1e-9, abs=1e-10)
    check.almost_equal(drive.amplitude.max(), 0.375, abs=1e-4)

    check.equal(drive.detuning.duration, drive.duration)
    check.almost_equal(drive.detuning.min(), -1.5, abs=1e-4)
    check.almost_equal(drive.detuning.max(), 1.5, abs=1e-4)


@pytest.mark.usefixtures("restore_rng_state")
def test_shaper_does_not_overflow_device() -> None:
    data_graph = DataGraph.triangular(4, 4, 1)
    np.random.seed(0)
    removed = np.random.choice(
        data_graph.number_of_nodes(), data_graph.number_of_nodes() - 8, replace=False
    )
    data_graph.remove_nodes_from(removed)
    coords = [data_graph.coords[n] for n in data_graph.nodes]
    dist_matrix = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        qubo = 1.0 / dist_matrix**6
    np.fill_diagonal(qubo, -1.0)
    instance = Instance(matrix.tensor(np.asarray(qubo, dtype=float)))

    device = qoolqit.AnalogDeviceWithDMM()
    register = embedding.greedy_layout.embed_for_device(instance, device)
    drive = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device)
    program = solving.analog_quantum_sampling.compile(register, drive, device)
    solution = Solution.from_results(LocalEmulator().run(program).results(), instance)

    assert solution.bitstrings.numel() > 0


@pytest.mark.duration(2)
@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
def test_proportional_diagonal_register_and_drive_shape_normalization(
    embedding_method: Literal["greedy_layout", "blade"],
) -> None:
    """The proportional-diagonal drive shaper must normalize the pulse consistently.

    It must use the same convention as the register.
    """
    qubo = np.array([[-1.0, 2.0], [2.0, -1.0]])
    instance = Instance(matrix.tensor(qubo))
    device = qoolqit.AnalogDeviceWithDMM()
    embedder = getattr(embedding, embedding_method)
    register = embedder.embed_for_device(instance, device)
    drive = drive_shaping.proportional_diagonal.build_drive(
        instance, register, device=device, kappa=0.25
    )

    interaction = register.interactions()[(0, 1)]
    detuning = drive.detuning(drive.detuning.duration)
    ratio = float(interaction / detuning)

    # Physically expected ratio: Q_ij encoded by the interaction, -0.5*Q_ii by
    # the final detuning -> 2.0 / (0.5) = 4.0
    expected = qubo[0, 1] / (-0.5 * qubo[0, 0])
    np.testing.assert_allclose(ratio, expected, rtol=1e-3)
