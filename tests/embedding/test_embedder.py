from __future__ import annotations

import pytest
from typing import Literal

from qoolqit import Device
from qubosolver import (
    Solver,
    Instance,
    SolverConfig,
    QuantumSolvingConfig,
    EmbeddingConfig,
)


@pytest.mark.priority(40)
@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
def test_embeddings_different_devices(
    qubo_for_testing_many_devices: Instance, local_device: Device, embedding_method: Literal["greedy_layout", "blade"],
) -> None:
    config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm=embedding_method, greedy_layout_traps="device"),
            device=local_device,
        ),
        do_postprocessing=False,
        do_preprocessing=False,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    assert solver._embedding()
