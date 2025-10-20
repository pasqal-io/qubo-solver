from __future__ import annotations

import pytest
import torch

import numpy as np
from qoolqit.devices import Device, DigitalAnalogDevice, AnalogDevice
from qubosolver import QUBOInstance
from qubosolver.config import (
    EmbeddingConfig,
    SolverConfig,
)
from qubosolver.pipeline.embedder import GreedyEmbedder, get_embedder
from qubosolver.solver import QuboSolver


devices: list[Device] = [AnalogDevice(), DigitalAnalogDevice()]


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


def test_greedy_embedder(qubo_instance_for_embedding: QUBOInstance) -> None:
    assert qubo_instance_for_embedding.size is not None
    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(
            embedding_method="greedy", greedy_traps=qubo_instance_for_embedding.size
        ),
    )
    solver = QuboSolver(qubo_instance_for_embedding, config)
    positions = solver.embedding()

    expected_greedy_positions = (
        torch.tensor(
            [[2.0000, 3.4641], [0.0000, 0.0000], [-2.0000, 3.4641], [4.0000, 0.0000]],
            dtype=torch.float16,
        )
        / solver.device.converter.factors[2]
    )
    expected_greedy_positions = expected_greedy_positions.tolist()

    assert len(positions.qubits) == len(expected_greedy_positions)

    for qubit_id, coordinate in enumerate(positions.qubits.values()):
        x, y = coordinate.clone().detach().to(dtype=torch.float16).tolist()
        x_, y_ = expected_greedy_positions[qubit_id]
        assert np.allclose(x, x_, atol=1e-3) and np.allclose(y, y_, atol=1e-3)


def test_greedy_max_radial_distance_constraint(
    qubo_instance_for_embedding: QUBOInstance,
) -> None:
    assert qubo_instance_for_embedding.size is not None

    for device in devices:
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                greedy_traps=qubo_instance_for_embedding.size,
                greedy_spacing=device._device.max_radial_distance,
            ),
            device=device,
        )

        solver = QuboSolver(qubo_instance_for_embedding, greedy_config)

        with pytest.raises(ValueError):
            solver.embedding()


def test_greedy_max_radial_distance_constraint_with_extra_greedy_traps(
    qubo_instance_for_embedding: QUBOInstance,
) -> None:
    assert qubo_instance_for_embedding.size is not None

    expected_greedy_positions = [
        torch.tensor(
            [
                [0.0000, 0.0000],
                [-9.5000, -16.4531],
                [-19.0000, 0.0000],
                [-9.5000, 16.4531],
            ],
            dtype=torch.float16,
        ).tolist(),
        torch.tensor(
            [
                [12.5000, -21.6562],
                [0.0000, 0.0000],
                [-25.0000, 0.0000],
                [-12.5000, -21.6562],
            ],
            dtype=torch.float16,
        ).tolist(),
    ]

    for scenario_idx, device in enumerate(devices):
        conv = device.converter.factors[2]
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                greedy_traps=qubo_instance_for_embedding.size * 2,
                greedy_spacing=device._device.max_radial_distance / 2,
            ),
            device=device,
        )

        solver = QuboSolver(qubo_instance_for_embedding, greedy_config)
        geometry = solver.embedding()

        assert len(geometry.qubits) == len(expected_greedy_positions[scenario_idx])

        for qubit_id, coordinate in enumerate(geometry.qubits.values()):
            x, y = coordinate.clone().detach().to(dtype=torch.float16).tolist()
            x_, y_ = expected_greedy_positions[scenario_idx][qubit_id]
            x_ /= conv
            y_ /= conv
            assert np.allclose(x, x_, atol=1e-3) and np.allclose(y, y_, atol=1e-3)
