# Copyright (c) Pasqal
#
# Licensed under the Pasqal Open-Source Software License (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import Literal

import pytest
from qoolqit import Device

from qubosolver import (
    EmbeddingConfig,
    Instance,
    QuantumSolvingConfig,
    Solver,
    SolverConfig,
)


@pytest.mark.duration(40)
@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
def test_embeddings_different_devices(
    qubo_for_testing_many_devices: Instance,
    local_device: Device,
    embedding_method: Literal["greedy_layout", "blade"],
) -> None:
    config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm=embedding_method, blade_steps_per_round=10),
            device=local_device,
        ),
        postprocessing=False,
        preprocessing=False,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    assert solver._embedding()
