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
    embedding,
    matrix,
)


@pytest.mark.priority(40)
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


def test_blade_embedder_forwards_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_embed(instance: Instance, *, config: embedding.blade.Config) -> None:
        captured["config"] = config

    monkeypatch.setattr(embedding.blade, "embed", fake_embed)

    qubo = matrix.as_tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    config = SolverConfig(
        solving=QuantumSolvingConfig(embedding=EmbeddingConfig(blade_steps_per_round=100)),
        postprocessing=False,
        preprocessing=False,
    )
    Solver(Instance(qubo), config)._embedding()

    blade_config = captured["config"]
    assert isinstance(blade_config, embedding.blade.Config)
    assert blade_config.steps_per_round == 100
