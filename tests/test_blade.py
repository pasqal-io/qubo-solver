import pytest

from qubosolver.algorithms.blade.blade import update_positions, em_blade
from pulser.devices import AnalogDevice
import numpy as np
import networkx as nx
import dataclasses


device = dataclasses.replace(
    AnalogDevice,
    rydberg_level=70,
    max_radial_distance=50,
    min_atom_distance=4,
)


@pytest.mark.parametrize(
    "max_distance_to_walk, expected_distance",
    [
        (np.inf, device.rydberg_blockade_radius(10)),
        (0, 1),
        (1, 3),
        (20, device.rydberg_blockade_radius(10)),
    ],
)
def test_update_positions(
    max_distance_to_walk: float | int, expected_distance: float | int
) -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    weight = 10
    qubo_graph.add_edge(0, 1, weight=weight)

    new_positions = update_positions(
        positions=np.array([[0, 0], [1, 0]]),
        qubo_graph=qubo_graph,
        device=device,
        max_distance_to_walk=max_distance_to_walk,
    )

    assert np.isclose(np.linalg.norm(new_positions[0] - new_positions[1]), expected_distance)


def test_max_dist_constraint() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=device.rabi_from_blockade(10 * np.sqrt(2)))

    new_positions = update_positions(
        positions=np.array([[-10, 0], [0, 10]]),
        qubo_graph=qubo_graph,
        device=device,
        max_dist=10,
    )

    assert np.isclose(np.linalg.norm(new_positions[0] - new_positions[1]), 10, rtol=1e-2)


def test_min_dist_constraint() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=device.rabi_from_blockade(10 * np.sqrt(2)))

    new_positions = update_positions(
        positions=np.array([[-10, 0], [0, 10]]),
        qubo_graph=qubo_graph,
        device=device,
        min_dist=30,
    )

    assert np.isclose(
        np.linalg.norm(new_positions[0] - new_positions[1]), 30, rtol=1e-2
    ), f"{np.linalg.norm(new_positions[0] - new_positions[1])} != 30"


def test_min_dist_constraint_limited() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=device.rabi_from_blockade(1))

    new_positions = update_positions(
        positions=np.array([[-1, 0], [1, 0]]),
        qubo_graph=qubo_graph,
        device=device,
        min_dist=10,
        max_distance_to_walk=(0, 2, 0),
    )

    assert (new_positions == np.array([[-3, 0], [3, 0]])).all()


def test_max_dist_constraint_limited() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=device.rabi_from_blockade(1))

    new_positions = update_positions(
        positions=np.array([[-10, 0], [10, 0]]),
        qubo_graph=qubo_graph,
        device=device,
        max_dist=1,
        max_distance_to_walk=(0, 0, 1),
    )

    assert (new_positions == np.array([[-9, 0], [9, 0]])).all()


def test_force_based_embedding() -> None:
    local_device = dataclasses.replace(
        device,
        max_radial_distance=4 * device.min_atom_distance,
        pre_calibrated_layouts=tuple(),
    )
    min_dist = local_device.min_atom_distance
    max_dist = local_device.max_radial_distance

    factor_dist_0_1 = 1 / 1.1
    factor_dist_2_3 = 1.2

    qubo = np.array(
        [
            [0, local_device.rabi_from_blockade(min_dist * factor_dist_0_1), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, local_device.rabi_from_blockade(min_dist * factor_dist_2_3)],
            [0, 0, 0, 0],
        ]
    )

    positions = em_blade(
        qubo=qubo,
        device=local_device,
        steps_per_round=1000,
        starting_positions=np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * max_dist / 3,
        dimensions=[2, 2],
        enforce_min_max_dist_ratio=True,
    )

    new_min_dist = np.linalg.norm(positions[0] - positions[1])
    new_max_dist = new_min_dist * (max_dist / min_dist)

    assert np.isclose(
        np.linalg.norm(positions[0] - positions[1]), new_min_dist
    ), f"{np.linalg.norm(positions[0] - positions[1])} != {new_min_dist}"
    assert (
        (new_max_dist - new_min_dist) < np.linalg.norm(positions[0] - positions[2]) < new_max_dist
    )
    assert new_max_dist - new_min_dist < np.linalg.norm(positions[0] - positions[3]) < new_max_dist

    assert new_max_dist - new_min_dist < np.linalg.norm(positions[1] - positions[2]) < new_max_dist
    assert new_max_dist - new_min_dist < np.linalg.norm(positions[1] - positions[3]) < new_max_dist

    assert np.isclose(
        np.linalg.norm(positions[2] - positions[3]),
        new_min_dist * factor_dist_2_3 / factor_dist_0_1,
        rtol=1e-1,
    )
