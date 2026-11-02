# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check
import qoolqit
import torch
from shapely.geometry import Point, Polygon

from qubosolver import Instance, embedding, matrix, torch_rng, vector
from qubosolver.embedding._algorithms.geometric import (
    WeightedZone,
    bfgs_placement,
    check_prohibited_zones,
    compute_distance_interaction_matrix,
    cost_interaction_point_continuous,
    random_point_in_geometry,
    vertices_to_place,
)
from qubosolver.utils._random import manual_seed
from qubosolver.utils.quantum import _max_min_distance_ratio


def _coords(register: qoolqit.Register) -> torch.Tensor:
    return torch.stack([torch.as_tensor(register.qubits[q]) for q in register.qubits_ids]).double()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_returns_a_subset_with_local_labels(decomposable_qubo: Instance, seed: int) -> None:
    register, variables = embedding.geometric.embed(decomposable_qubo, rng=torch_rng(seed))

    check.greater_equal(len(variables), 1)
    check.less_equal(len(variables), decomposable_qubo.size)
    check.equal(len(set(variables)), len(variables))
    check.equal(variables, sorted(variables))
    check.is_true(all(0 <= v < decomposable_qubo.size for v in variables))
    check.equal(list(register.qubits_ids), list(range(len(variables))))


def test_embed_is_reproducible_with_the_global_seed(decomposable_qubo: Instance) -> None:
    manual_seed(5)
    register1, variables1 = embedding.geometric.embed(decomposable_qubo)
    manual_seed(5)
    register2, variables2 = embedding.geometric.embed(decomposable_qubo)

    check.equal(variables1, variables2)
    torch.testing.assert_close(_coords(register1), _coords(register2))


@pytest.mark.parametrize(
    "instance",
    [
        Instance(matrix.tensor([[-1.0]])),
        Instance(matrix.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])),
    ],
    ids=["single", "weakly_coupled"],
)
def test_embed_places_at_least_one_atom(instance: Instance) -> None:
    register, variables = embedding.geometric.embed(instance, rng=torch_rng(0))

    check.greater_equal(len(variables), 1)
    check.equal(len(register), len(variables))


def test_embed_for_device_single_atom() -> None:
    instance = Instance(matrix.tensor([[-1.0]]))
    register, variables = embedding.geometric.embed_for_device(
        instance, qoolqit.AnalogDeviceWithDMM(), rng=torch_rng(0)
    )

    check.equal(variables, [0])
    check.equal(len(register), 1)


def test_embed_rejects_diagonal_instance() -> None:
    instance = Instance(matrix.tensor([[-1.0, 0.0], [0.0, -2.0]]))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        embedding.geometric.embed(instance, rng=torch_rng(0))


def test_embed_for_device_rejects_diagonal_instance() -> None:
    instance = Instance(matrix.tensor([[-1.0, 0.0], [0.0, -2.0]]))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        embedding.geometric.embed_for_device(
            instance, qoolqit.AnalogDeviceWithDMM(), rng=torch_rng(0)
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_for_device_respects_the_distance_ratio(
    decomposable_qubo: Instance, seed: int
) -> None:
    device = qoolqit.AnalogDeviceWithDMM()

    register, variables = embedding.geometric.embed_for_device(
        decomposable_qubo, device, rng=torch_rng(seed)
    )

    coords = _coords(register)
    min_distance = torch.pdist(coords).min()
    max_radial_distance = coords.norm(dim=1).max()
    check.less_equal(
        (max_radial_distance / min_distance).item(), _max_min_distance_ratio(device) + 1e-6
    )


def test_config_from_device() -> None:
    device = qoolqit.AnalogDevice()
    config = embedding.geometric.Config.from_device(device)
    check.equal(config.max_min_dist_ratio, _max_min_distance_ratio(device))


def test_embed_rejects_empty_instance() -> None:
    with pytest.raises(ValueError, match="empty"):
        embedding.geometric.embed(Instance())


def test_embed_rejects_negative_off_diagonal() -> None:
    instance = Instance(matrix.tensor([[-1.0, -1.0], [-1.0, -1.0]]))
    with pytest.raises(ValueError, match="negative off-diagonal"):
        embedding.geometric.embed(instance)


@pytest.mark.parametrize("diagonal", [0.0, 1.0])
def test_embed_rejects_non_negative_diagonal(diagonal: float) -> None:
    instance = Instance(matrix.tensor([[-1.0, 2.0], [2.0, diagonal]]))
    with pytest.raises(ValueError, match=r"Variables \[1\] .*apply_recursively"):
        embedding.geometric.embed(instance)


def test_vertices_to_place_dimensions(decomposable_qubo: Instance) -> None:
    qubo_mat = decomposable_qubo.matrix
    size = decomposable_qubo.size

    dist_matrix = compute_distance_interaction_matrix(qubo_mat)
    check.equal(dist_matrix.shape, qubo_mat.shape)
    check.is_true(torch.all(torch.diag(dist_matrix) == torch.diag(qubo_mat)))

    vertices = vertices_to_place(dist_matrix, qubo_mat)
    check.equal(len(vertices), size)
    for vertex in vertices.values():
        check.less_equal(len(vertex["blocking_vertices"]), size)
        check.less_equal(len(vertex["separated_vertices"]), size)
        check.less_equal(len(vertex["neighbors_id"]), size)
        check.equal(len(vertex["neighbors_id"]), len(vertex["neighbors_weight"]))


def test_compute_distance_interaction_matrix_zero_output() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0

    Q = matrix.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )

    torch.testing.assert_close(dist_matrix, torch.zeros_like(Q))


def test_compute_distance_interaction_diagonal() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0

    Q = matrix.tensor(
        [
            [-10, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0],
        ],
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )
    expected_dist_matrix = neglecting_inter_distance * torch.ones_like(Q)
    expected_dist_matrix.diagonal().copy_(Q.diag())

    torch.testing.assert_close(dist_matrix, expected_dist_matrix)


def test_random_point_in_geometry_rejects_non_positive_max_trials() -> None:
    with pytest.raises(ValueError, match="max_trials must be a positive integer"):
        random_point_in_geometry(Point(0, 0).buffer(1.0), max_trials=0, rng=torch_rng(1321356387))


def test_random_point_in_geometry_gives_up_after_max_trials() -> None:
    # A zero-area polygon contains no point, so every trial fails.
    degenerate = Polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    p = random_point_in_geometry(degenerate, max_trials=3, rng=torch_rng(1321356387))
    check.is_false(degenerate.contains(Point(p[0].item(), p[1].item())))


def test_cost_interaction_point_continuous_blocked_indices() -> None:
    def cost(blocked_indices: list[int]) -> float:
        # Interaction 1 / 1**6 = 1 at distance 1 from the placed point, target 0.5.
        return cost_interaction_point_continuous(
            np.array([1.0, 0.0]), [[0.0, 0.0]], [0.5], blocked_indices, 0.1, 10.0
        )

    # Unblocked, the cost is the squared mismatch; blocked, an interaction
    # stronger than the target is not penalized.
    check.almost_equal(cost([]), 0.25)
    check.almost_equal(cost([0]), 0.0)


def test_bfgs_placement_with_blocking_vertex() -> None:
    placed_vertices = {0: WeightedZone(0, 0.0, 0.0, -1.0, 0.1)}
    qubo = matrix.tensor([[-1.0, 0.5], [0.5, -1.0]])
    vertices = {1: {"blocking_vertices": torch.tensor([0])}}
    result = bfgs_placement(
        1, vector.tensor([1.0, 0.0]), placed_vertices, qubo, vertices, 0.1, 10.0
    )
    check.less_equal(result.fun, 1e-6)


def test_check_prohibited_zones() -> None:
    placed_vertices = {0: WeightedZone(0, 0.0, 0.0, -1.0, 0.3)}
    # Inside the forbidden zone.
    check.is_false(check_prohibited_zones(placed_vertices, Point(0.2, 0.0)))
    # Outside the forbidden zone but too close to the placed vertex.
    check.is_false(check_prohibited_zones(placed_vertices, Point(0.4, 0.0)))
    check.is_true(check_prohibited_zones(placed_vertices, Point(1.0, 0.0)))
