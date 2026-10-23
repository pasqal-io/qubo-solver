from __future__ import annotations

import pytest
import pytest_check as check
import qoolqit
import torch

from qubosolver import Instance, embedding, matrix, torch_rng
from qubosolver.embedding._algorithms.geometric import (
    compute_distance_interaction_matrix,
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
        Instance(matrix.tensor([[-1.0, 0.0], [0.0, -2.0]])),
        Instance(matrix.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])),
    ],
    ids=["single", "uncoupled", "weakly_coupled"],
)
def test_embed_places_at_least_one_atom(instance: Instance) -> None:
    register, variables = embedding.geometric.embed(instance, rng=torch_rng(0))

    check.greater_equal(len(variables), 1)
    check.equal(len(register), len(variables))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_embed_for_device_respects_the_distance_ratio(
    decomposable_qubo: Instance, seed: int
) -> None:
    device = qoolqit.AnalogDeviceWithDMM()

    register, variables = embedding.geometric.embed_for_device(
        decomposable_qubo, device, rng=torch_rng(seed)
    )
    if len(variables) < 2:
        pytest.skip("Only one atom placed.")

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
