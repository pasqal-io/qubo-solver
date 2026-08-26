from __future__ import annotations

import pytest

from qoolqit import Device
from qubosolver import (
    Solver,
    Instance,
    embedding,
    solvers,
)
from qubosolver.embedding._embedder import GreedyEmbedder, _get_embedder


@pytest.mark.priority(40)
@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
def test_embeddings_different_devices(
    qubo_for_testing_many_devices: Instance, local_device: Device, embedding_method: str
) -> None:
    config = solvers.Config(
        solving=solvers.quantum.Config(
            embedding=embedding.Config(algorithm=embedding_method, greedy_layout_traps="device"),
            device=local_device,
        ),
        do_postprocessing=False,
        do_preprocessing=False,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    assert solver._embedding()
