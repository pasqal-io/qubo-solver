from __future__ import annotations

import pytest
import torch

from qoolqit import Device, DigitalAnalogDevice, AnalogDevice, AnalogDeviceWithDMM
from qubosolver import (
    Solver,
    Instance,
    EmbeddingConfig,
    SolverConfig,
    tensor,
    vector,
    Vector,
)
from qubosolver.embedding._embedder import GreedyEmbedder, _get_embedder


@pytest.mark.priority(40)
@pytest.mark.parametrize("embedding_method", ["greedy", "blade"])
def test_embeddings_different_devices(
    qubo_for_testing_many_devices: Instance, local_device: Device, embedding_method: str
) -> None:
    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(embedding_method=embedding_method, greedy_traps=-1),
        do_postprocessing=False,
        do_preprocessing=False,
        device=local_device,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    assert solver.embedding()


def test_custom_embedder(simple_qubo_instance: Instance) -> None:

    class MockGreedyEmbedder(GreedyEmbedder):
        pass

    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(embedding_method=MockGreedyEmbedder),
    )
    backend = config.backend
    shaper = _get_embedder(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockGreedyEmbedder)


def test_correctness_greedy_embedder(qubo_instance_for_embedding: Instance) -> None:
    assert qubo_instance_for_embedding.size is not None
    config = SolverConfig(
        use_quantum=True,
        embedding=EmbeddingConfig(
            embedding_method="greedy",
            greedy_traps=qubo_instance_for_embedding.size,
            greedy_spacing=4.0,
        ),
        device=DigitalAnalogDevice(),
    )
    solver = Solver(qubo_instance_for_embedding, config)
    positions = solver.embedding()

    expected_greedy_positions = (
        tensor.tensor(
            [[-2.0000, 3.4641], [0.0000, 0.0000], [2.0000, 3.4641], [4.0000, 0.0000]],
        )
        / solver.device.converter.factors[2]
    )

    assert len(positions.qubits) == len(expected_greedy_positions)

    def to_tuple(v: Vector) -> tuple[float, float]:
        return v[0].item(), v[1].item()

    for qubit_id, coordinate in enumerate(sorted(positions.qubits.values(), key=to_tuple)):
        p = vector.from_torch(coordinate.detach().clone())
        expected_p = expected_greedy_positions[qubit_id]
        torch.testing.assert_close(p, expected_p, atol=1e-3, rtol=1e-3)


def test_error_greedy_max_radial_distance_constraint(
    qubo_instance_for_embedding: Instance,
) -> None:
    assert qubo_instance_for_embedding.size is not None

    for device in [AnalogDevice(), AnalogDeviceWithDMM()]:
        max_radial_distance = device.specs["max_radial_distance"]
        assert max_radial_distance is not None
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                greedy_traps=qubo_instance_for_embedding.size,
                greedy_spacing=max_radial_distance,
            ),
            device=device,
        )

        solver = Solver(qubo_instance_for_embedding, greedy_config)
        # Setting a spacing larger than the max_radial_distance is not an error,
        # since scaling is performed
        solver.embedding()


@pytest.mark.parametrize("normalized", [True, False], ids=["normalized", "not_normalized"])
def test_correctness_greedy_max_radial_distance_constraint_with_extra_greedy_traps(
    qubo_instance_for_embedding: Instance,
    normalized: bool,
) -> None:
    assert qubo_instance_for_embedding.size is not None

    expected = tensor.tensor(
        [
            [0.0000, 0.0000],
            [-9.5000, -16.4531],
            [-19.0000, 0.0000],
            [-9.5000, 16.4531],
        ],
    )
    # Solutions are valid up to isometry
    if tensor.dtype() == torch.float64:
        expected[3] *= -1.0

    def to_tuple(v: Vector) -> tuple[float, float]:
        assert v.numel() == 2
        return v[0].item(), v[1].item()

    for device in [AnalogDevice(), AnalogDeviceWithDMM()]:

        min_distance = 1.0 if normalized else None

        assert device._device.max_radial_distance is not None
        greedy_config = SolverConfig(
            use_quantum=True,
            embedding=EmbeddingConfig(
                embedding_method="greedy",
                greedy_traps=qubo_instance_for_embedding.size * 2,
                greedy_spacing=device._device.max_radial_distance / 2.0,
                min_distance=min_distance,
            ),
            device=device,
        )

        solver = Solver(qubo_instance_for_embedding, greedy_config)
        geometry = solver.embedding()

        assert len(geometry.qubits) == len(expected)
        if normalized:
            conv = torch.cdist(expected, expected).fill_diagonal_(float("inf")).min().item()
        else:
            conv = device.converter.factors[2]

        sorted_expected = sorted(expected, key=to_tuple)

        for qubit_id, coordinate in enumerate(sorted(geometry.qubits.values(), key=to_tuple)):
            p = vector.from_torch(coordinate.detach().clone())
            expected_p = sorted_expected[qubit_id] / conv
            torch.testing.assert_close(p, expected_p, atol=1e-3, rtol=1e-3)
