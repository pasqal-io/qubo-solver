from __future__ import annotations

import pytest
import torch
from qoolqit._solvers import get_backend

from qubosolver import QUBOInstance
from qubosolver.config import (
    BackendConfig,
    DeviceType,
    EmbeddingConfig,
    SolverConfig,
)
from qubosolver.pipeline.embedder import GreedyEmbedder, get_embedder
from qubosolver.solver import QuboSolver


def test_custom_embedder(simple_qubo_instance: QUBOInstance) -> None:

    class MockGreedyEmbedder(GreedyEmbedder):
        pass

    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(embedding_method=MockGreedyEmbedder),
    )
    backend = get_backend(config.backend_config)
    shaper = get_embedder(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockGreedyEmbedder)


def test_greedy_embedder(qubo_instance_for_embedding: QUBOInstance) -> None:
    assert qubo_instance_for_embedding.size is not None
    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(
            embedding_method="greedy", traps=qubo_instance_for_embedding.size
        ),
    )
    solver = QuboSolver(qubo_instance_for_embedding, config)
    positions = solver.embedding()

    expected_greedy_positions = torch.tensor(
        [[2.0000, 3.4641], [0.0000, 0.0000], [-2.0000, 3.4641], [4.0000, 0.0000]],
        dtype=torch.float16,
    ).tolist()

    assert len(positions.register.qubits) == len(expected_greedy_positions)

    for qubit_id, coordinate in enumerate(positions.register.qubits.values()):
        x, y = coordinate.as_tensor().clone().detach().to(dtype=torch.float16).tolist()
        x_, y_ = expected_greedy_positions[qubit_id]
        assert (x == x_) and (y == y_)


devices: list[DeviceType] = [DeviceType.ANALOG_DEVICE, DeviceType.DIGITAL_ANALOG_DEVICE]


def test_greedy_max_radial_distance_constraint(
    qubo_instance_for_embedding: QUBOInstance,
) -> None:
    assert qubo_instance_for_embedding.size is not None

    for device in devices:
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                traps=qubo_instance_for_embedding.size,
                spacing=device.value.max_radial_distance,
            ),
            backend_config=BackendConfig(device=device),
        )

        solver = QuboSolver(qubo_instance_for_embedding, greedy_config)

        with pytest.raises(ValueError):
            solver.embedding()


def test_greedy_max_radial_distance_constraint_with_extra_traps(
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
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                traps=qubo_instance_for_embedding.size * 2,
                spacing=device.value.max_radial_distance / 2,
            ),
            backend_config=BackendConfig(device=device),
        )

        solver = QuboSolver(qubo_instance_for_embedding, greedy_config)
        geometry = solver.embedding()

        assert len(geometry.register.qubits) == len(expected_greedy_positions[scenario_idx])

        for qubit_id, coordinate in enumerate(geometry.register.qubits.values()):
            x, y = coordinate.as_tensor().clone().detach().to(dtype=torch.float16).tolist()
            x_, y_ = expected_greedy_positions[scenario_idx][qubit_id]
            assert (x == x_) and (y == y_)
