import pytest

from qubosolver.algorithms.blade.blade import update_positions, em_blade, em_blade_for_device
from qubosolver.algorithms.blade._helpers import normalized_best_dist, normalized_interaction
from pulser.devices import AnalogDevice
import numpy as np
import networkx as nx
import dataclasses


@pytest.mark.parametrize(
    "max_distance_to_walk, expected_distance",
    [
        (np.inf, normalized_best_dist(1e-4)),
        (0, 1),
        (1, 3),
        (20, normalized_best_dist(1e-4)),
    ],
)
def test_update_positions(
    max_distance_to_walk: float | int, expected_distance: float | int
) -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    weight = 1e-4
    qubo_graph.add_edge(0, 1, weight=weight)

    new_positions = update_positions(
        positions=np.array([[0, 0], [1, 0]]),
        qubo_graph=qubo_graph,
        max_distance_to_walk=max_distance_to_walk,
    )

    assert np.isclose(np.linalg.norm(new_positions[0] - new_positions[1]), expected_distance)


def test_max_dist_constraint() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=1)

    max_radial_dist = 0.1

    new_positions = update_positions(
        positions=np.array([[-0.5, 0], [0.5, 0]]),
        qubo_graph=qubo_graph,
        max_dist=max_radial_dist,
    )

    assert np.isclose(
        np.linalg.norm(new_positions[0] - new_positions[1]), 2 * max_radial_dist, rtol=1e-2
    )


def test_min_dist_constraint() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=normalized_interaction(10 * np.sqrt(2)))

    new_positions = update_positions(
        positions=np.array([[-10, 0], [0, 10]]),
        qubo_graph=qubo_graph,
        min_dist=30,
    )

    assert np.isclose(
        np.linalg.norm(new_positions[0] - new_positions[1]), 30, rtol=1e-2
    ), f"{np.linalg.norm(new_positions[0] - new_positions[1])} != 30"


def test_min_dist_constraint_limited() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=normalized_interaction(1))

    new_positions = update_positions(
        positions=np.array([[-1, 0], [1, 0]]),
        qubo_graph=qubo_graph,
        min_dist=10,
        max_distance_to_walk=(0, 2, 0),
    )

    assert (new_positions == np.array([[-3, 0], [3, 0]])).all()


def test_max_dist_constraint_limited() -> None:
    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=normalized_interaction(1))

    new_positions = update_positions(
        positions=np.array([[-10, 0], [10, 0]]),
        qubo_graph=qubo_graph,
        max_dist=1,
        max_distance_to_walk=(0, 0, 1),
    )

    assert (new_positions == np.array([[-9, 0], [9, 0]])).all()


def test_force_based_embedding() -> None:
    min_dist = 1
    max_dist = 2

    factor_dist_0_1 = 1 / 1.1
    factor_dist_2_3 = 1.2

    qubo = np.array(
        [
            [0, normalized_interaction(min_dist * factor_dist_0_1), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, normalized_interaction(min_dist * factor_dist_2_3)],
            [0, 0, 0, 0],
        ]
    )

    positions = em_blade(
        qubo=qubo,
        max_min_dist_ratio=max_dist / min_dist,
        steps_per_round=1000,
        starting_positions=np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * max_dist / 3,
        dimensions=[2, 2],
    )

    new_min_dist = np.linalg.norm(positions[0] - positions[1])
    new_max_dist = new_min_dist * (max_dist / min_dist)
    new_max_diameter_dist = 2 * new_max_dist

    assert np.isclose(
        np.linalg.norm(positions[0] - positions[1]), new_min_dist
    ), f"{np.linalg.norm(positions[0] - positions[1])} != {new_min_dist}"
    assert (
        (new_max_diameter_dist - new_min_dist)
        < np.linalg.norm(positions[0] - positions[2])
        < new_max_diameter_dist
    )
    assert (
        new_max_diameter_dist - new_min_dist
        < np.linalg.norm(positions[0] - positions[3])
        < new_max_diameter_dist
    )

    assert (
        new_max_diameter_dist - new_min_dist
        < np.linalg.norm(positions[1] - positions[2])
        < new_max_diameter_dist
    )
    assert (
        new_max_diameter_dist - new_min_dist
        < np.linalg.norm(positions[1] - positions[3])
        < new_max_diameter_dist
    )

    assert np.isclose(
        np.linalg.norm(positions[2] - positions[3]),
        new_min_dist * factor_dist_2_3 / factor_dist_0_1,
        rtol=1e-1,
    )


def test_high_dimension_increase_after_equilibrium() -> None:
    qubo = np.array(
        [
            [0.0, 0.7, 0.3, 0.5, 0.4, 0.9, 0.9, 0.7, 0.9, 0.8],
            [0.0, 0.0, 0.7, 0.4, 0.8, 0.4, 0.8, 1.0, 0.5, 0.8],
            [0.0, 0.0, 0.0, 0.7, 0.5, 0.8, 0.0, 0.8, 0.7, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.1, 0.9, 0.2, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.4, 0.7, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.9, 0.4, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    em_blade(qubo, dimensions=[2, 2, 10], steps_per_round=100)


def test_drawing() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qubo_graph = nx.Graph()
    qubo_graph.add_nodes_from([i for i in range(2)])
    qubo_graph.add_edge(0, 1, weight=normalized_interaction(1))

    plt.close("all")
    assert len(plt.get_fignums()) == 0
    update_positions(
        positions=np.array([[-10, 0], [10, 0]]),
        qubo_graph=qubo_graph,
        max_dist=1,
        max_distance_to_walk=(0, 0, 1),
        draw_step=True,
    )
    assert len(plt.get_fignums()) > 0
    plt.close("all")


def test_with_device() -> None:
    device = dataclasses.replace(
        AnalogDevice,
        rydberg_level=70,
        max_radial_distance=50,
        min_atom_distance=4,
    )

    qubo = np.array(
        [
            [0, 2],
            [0, 0],
        ]
    )
    positions = em_blade_for_device(qubo, device=device, dimensions=[2, 2], steps_per_round=100)
    distances = np.triu(
        np.linalg.norm(positions[np.newaxis, :, :] - positions[:, np.newaxis, :], axis=-1), k=1
    )

    def best_dist(weight: float) -> float:
        return device.rydberg_blockade_radius(weight)

    expected_distances = np.triu(np.vectorize(best_dist, signature="(m,n)->(m,n)")(qubo), k=1)
    assert np.allclose(distances, expected_distances)
