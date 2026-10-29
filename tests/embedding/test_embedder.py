# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import pytest_check as check

from qubosolver import (
    EmbeddingConfig,
    Instance,
    QuantumSolvingConfig,
    Solver,
    SolverConfig,
)
from qubosolver.solver._embedder import BLaDEmbedder, _get_embedder

if TYPE_CHECKING:
    from qoolqit import Device


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


def test_get_embedder_invalid_algorithm(simple_qubo_instance: Instance) -> None:
    config = QuantumSolvingConfig()
    config.embedding.algorithm = "dummy"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        _get_embedder(simple_qubo_instance, config, config.backend)


@pytest.mark.duration(6)
def test_blade_embedder_default_steps_per_round(simple_qubo_instance: Instance) -> None:
    config = QuantumSolvingConfig(embedding=EmbeddingConfig(blade_steps_per_round=None))
    register = BLaDEmbedder(simple_qubo_instance, config, config.backend).embed()
    check.equal(register.n_qubits, simple_qubo_instance.size)
