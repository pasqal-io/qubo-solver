from __future__ import annotations

import pytest

from qoolqit.devices import Device
from qubosolver import QUBOInstance
from qubosolver.config import (
    EmbeddingConfig,
    SolverConfig,
)
from qubosolver.pipeline.embedder import GreedyEmbedder, get_embedder
from qubosolver.solver import QuboSolver


@pytest.mark.priority(40)
@pytest.mark.parametrize("embedding_method", ["greedy", "blade"])
def test_embeddings_different_devices(
    qubo_for_testing_many_devices: QUBOInstance, local_device: Device, embedding_method: str
) -> None:
    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(embedding_method=embedding_method, greedy_traps=-1),
        do_postprocessing=False,
        do_preprocessing=False,
        device=local_device,
    )
    solver = QuboSolver(qubo_for_testing_many_devices, config)
    assert solver.embedding()


def test_custom_embedder(simple_qubo_instance: QUBOInstance) -> None:

    class MockGreedyEmbedder(GreedyEmbedder):
        pass

    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(embedding_method=MockGreedyEmbedder),
    )
    backend = config.backend
    shaper = get_embedder(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockGreedyEmbedder)
