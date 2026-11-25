# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check
import qoolqit
import scipy
from shapely.geometry import Point, Polygon

from qubosolver import Instance, embedding, numpy_rng
from qubosolver.embedding._algorithms.partial_greedy import (
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


def _coords(register: qoolqit.Register) -> np.ndarray:
    return np.array([register.qubits[q] for q in register.qubits_ids], dtype=np.float64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_returns_a_subset_with_local_labels(decomposable_qubo: Instance, seed: int) -> None:
    register, variables = embedding.partial_greedy.embed(decomposable_qubo, rng=numpy_rng(seed))

    check.greater_equal(len(variables), 1)
    check.less_equal(len(variables), decomposable_qubo.size)
    check.equal(len(set(variables)), len(variables))
    check.equal(variables, sorted(variables))
    check.is_true(all(0 <= v < decomposable_qubo.size for v in variables))
    check.equal(list(register.qubits_ids), list(range(len(variables))))


def test_embed_is_reproducible_with_the_global_seed(decomposable_qubo: Instance) -> None:
    manual_seed(5)
    register1, variables1 = embedding.partial_greedy.embed(decomposable_qubo)
    manual_seed(5)
    register2, variables2 = embedding.partial_greedy.embed(decomposable_qubo)

    check.equal(variables1, variables2)
    np.testing.assert_allclose(_coords(register1), _coords(register2), strict=True)


@pytest.mark.parametrize(
    "instance",
    [
        Instance(np.asarray([[-1.0]], dtype=np.float64)),
        Instance(np.asarray([[1.0]], dtype=np.float64)),
        Instance(
            np.asarray([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]], dtype=np.float64)
        ),
    ],
    ids=["single", "single_non_negative", "weakly_coupled"],
)
def test_embed_places_at_least_one_atom(instance: Instance) -> None:
    register, variables = embedding.partial_greedy.embed(instance, rng=numpy_rng(0))

    check.greater_equal(len(variables), 1)
    check.equal(len(register), len(variables))


def test_embed_for_device_single_atom() -> None:
    instance = Instance(np.asarray([[-1.0]], dtype=np.float64))
    register, variables = embedding.partial_greedy.embed_for_device(
        instance, qoolqit.AnalogDeviceWithDMM(), rng=numpy_rng(0)
    )

    check.equal(variables, [0])
    check.equal(len(register), 1)


def test_embed_rejects_diagonal_instance() -> None:
    instance = Instance(np.asarray([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        embedding.partial_greedy.embed(instance, rng=numpy_rng(0))


def test_embed_for_device_rejects_diagonal_instance() -> None:
    instance = Instance(np.asarray([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        embedding.partial_greedy.embed_for_device(
            instance, qoolqit.AnalogDeviceWithDMM(), rng=numpy_rng(0)
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_for_device_respects_the_distance_ratio(
    decomposable_qubo: Instance, seed: int
) -> None:
    device = qoolqit.AnalogDeviceWithDMM()

    register, variables = embedding.partial_greedy.embed_for_device(
        decomposable_qubo, device, rng=numpy_rng(seed)
    )
    check.equal(len(register), len(variables))
    assert len(variables) >= 2  # the distance ratio needs two atoms

    coords = _coords(register)
    min_distance = scipy.spatial.distance.pdist(coords).min()
    max_radial_distance = np.linalg.norm(coords, axis=1).max()
    check.less_equal(
        (max_radial_distance / min_distance).item(), _max_min_distance_ratio(device) + 1e-6
    )


def test_config_from_device() -> None:
    device = qoolqit.AnalogDevice()
    config = embedding.partial_greedy.Config.from_device(device)
    check.equal(config.max_min_dist_ratio, _max_min_distance_ratio(device))


def test_embed_rejects_empty_instance() -> None:
    with pytest.raises(ValueError, match="empty"):
        embedding.partial_greedy.embed(Instance())


def test_embed_rejects_negative_off_diagonal() -> None:
    instance = Instance(np.asarray([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="negative off-diagonal"):
        embedding.partial_greedy.embed(instance)


def test_embed_rejects_positive_diagonal() -> None:
    instance = Instance(np.asarray([[-1.0, 2.0], [2.0, 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match=r"Variables \[1\] .*apply_recursively"):
        embedding.partial_greedy.embed(instance)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_accepts_zero_diagonal(seed: int) -> None:
    # A zero diagonal coefficient has an infinite blocking radius.
    instance = Instance(np.asarray([[-1.0, 2.0], [2.0, 0.0]], dtype=np.float64))
    register, variables = embedding.partial_greedy.embed(instance, rng=numpy_rng(seed))
    check.is_in(variables, [[0], [1], [0, 1]])
    check.equal(len(register), len(variables))


def test_vertices_to_place_dimensions(decomposable_qubo: Instance) -> None:
    qubo_mat = decomposable_qubo.matrix
    size = decomposable_qubo.size

    dist_matrix = compute_distance_interaction_matrix(qubo_mat)
    check.equal(dist_matrix.shape, qubo_mat.shape)
    check.is_true(np.all(np.diag(dist_matrix) == np.diag(qubo_mat)))

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

    Q = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
        dtype=np.float64,
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )

    np.testing.assert_allclose(dist_matrix, np.zeros_like(Q), strict=True)


def test_compute_distance_interaction_diagonal() -> None:

    neglecting_inter_distance = 15.0
    neglecting_max_coefficient = 1.0

    Q = np.asarray(
        [
            [-10, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    dist_matrix = compute_distance_interaction_matrix(
        Q, neglecting_inter_distance, neglecting_max_coefficient
    )
    expected_dist_matrix = neglecting_inter_distance * np.ones_like(Q)
    np.fill_diagonal(expected_dist_matrix, np.diag(Q))

    np.testing.assert_allclose(dist_matrix, expected_dist_matrix, strict=True)


def test_random_point_in_geometry_rejects_non_positive_max_trials() -> None:
    with pytest.raises(ValueError, match="max_trials must be a positive integer"):
        random_point_in_geometry(Point(0, 0).buffer(1.0), max_trials=0, rng=numpy_rng(1321356387))


def test_random_point_in_geometry_gives_up_after_max_trials() -> None:
    # A zero-area polygon contains no point, so every trial fails.
    degenerate = Polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    p = random_point_in_geometry(degenerate, max_trials=3, rng=numpy_rng(1321356387))
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
    qubo = np.asarray([[-1.0, 0.5], [0.5, -1.0]], dtype=np.float64)
    vertices = {1: {"blocking_vertices": np.array([0])}}
    result = bfgs_placement(
        1, np.asarray([1.0, 0.0], dtype=np.float64), placed_vertices, qubo, vertices, 0.1, 10.0
    )
    check.less_equal(result.fun, 1e-6)


def test_check_prohibited_zones() -> None:
    placed_vertices = {0: WeightedZone(0, 0.0, 0.0, -1.0, 0.3)}
    # Inside the forbidden zone.
    check.is_false(check_prohibited_zones(placed_vertices, Point(0.2, 0.0)))
    # Outside the forbidden zone but too close to the placed vertex.
    check.is_false(check_prohibited_zones(placed_vertices, Point(0.4, 0.0)))
    check.is_true(check_prohibited_zones(placed_vertices, Point(1.0, 0.0)))


def _pair_qubo(q_ii: float, q_jj: float, q_ij: float) -> np.ndarray:
    return np.asarray([[q_ii, q_ij], [q_ij, q_jj]], dtype=np.float64)


# Under x^T Q x, setting both variables to 1 costs Q_ii + Q_jj + 2 Q_ij: it stops paying off
# against the best single variable when Q_ij > -max(Q_ii, Q_jj) / 2.
@pytest.mark.parametrize(
    ("q_ii", "q_jj", "q_ij", "expected_dist_ij"),
    [
        (-8.0, -8.0, 3.0, 3.0 ** (-1 / 6)),  # both-on pays off
        (-8.0, -8.0, 6.0, 0.0),  # -Q_ii / 2 < Q_ij < -Q_ii: both-on does not pay off
        (-8.0, -8.0, 10.0, 0.0),  # both-on does not pay off
        (-8.0, -12.0, 3.0, 3.0 ** (-1 / 6)),
        (-8.0, -12.0, 6.0, 0.0),
        (-8.0, -12.0, 13.0, 0.0),
    ],
)
def test_compute_distance_interaction_matrix_blocks_iff_both_on_does_not_pay_off(
    q_ii: float, q_jj: float, q_ij: float, expected_dist_ij: float
) -> None:
    dist_matrix = compute_distance_interaction_matrix(_pair_qubo(q_ii, q_jj, q_ij))

    check.almost_equal(dist_matrix[0, 1].item(), expected_dist_ij)
    check.almost_equal(dist_matrix[1, 0].item(), expected_dist_ij)


# WeightedZone.radius2 and the blocking threshold of compute_distance_interaction_matrix must
# agree: a vertex at the interaction distance of a pair lies in the blocking disc of the other
# iff the matrix marks the pair as blocking.
@pytest.mark.parametrize("q_ij", [3.0, 6.0, 10.0])
def test_weighted_zone_blocking_radius_matches_distance_matrix(q_ij: float) -> None:
    weight = -8.0
    dist_matrix = compute_distance_interaction_matrix(_pair_qubo(weight, weight, q_ij))
    matrix_blocks = bool(dist_matrix[0, 1] == 0)

    zone = WeightedZone(0, 0.0, 0.0, weight, min_distance=0.1)
    interaction_distance = q_ij ** (-1 / 6)
    zone_blocks = zone.circle_end_forbidden.contains(Point(interaction_distance, 0.0))

    check.equal(zone_blocks, matrix_blocks)


@pytest.mark.parametrize("max_radial_distance", [2.0, np.inf])
def test_weighted_zone_with_zero_weight_blocks_the_search_disc(max_radial_distance: float) -> None:
    # Placed at the edge of the search disc, the vertex still blocks the opposite edge.
    search_radius = 2.0 if max_radial_distance == 2.0 else 50.0
    zone = WeightedZone(
        0, search_radius, 0.0, 0.0, min_distance=0.1, max_radial_distance=max_radial_distance
    )

    check.equal(zone.radius2, float("inf"))
    check.is_true(zone.circle_end_forbidden.contains(Point(-search_radius, 0.0)))
    check.is_true(zone.range_zone.is_empty)
