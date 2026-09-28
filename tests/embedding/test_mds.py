from __future__ import annotations

import numpy as np
import pytest_check as check

from qubosolver.embedding._algorithms.mds import embed_mds, reduce_distances_by_shortest_paths


def test_embed_mds_shape() -> None:
    rng = np.random.default_rng(0)
    qubo = rng.random((6, 6))
    qubo = qubo + qubo.T
    positions = embed_mds(qubo)
    check.equal(positions.shape, (6, 2))
    check.is_true(np.isfinite(positions).all())


def test_embed_mds_disconnected() -> None:
    # Two components with no coupling between them.
    qubo = np.zeros((4, 4))
    qubo[0, 1] = qubo[1, 0] = 1.0
    qubo[2, 3] = qubo[3, 2] = 1.0
    positions = embed_mds(qubo)
    check.equal(positions.shape, (4, 2))
    check.is_true(np.isfinite(positions).all())


def path_distances() -> np.ndarray:
    # Path 0 - 1 - 2 with a long direct 0 - 2 edge, and 3 only linked to 2.
    inf = np.inf
    return np.array(
        [
            [0.0, 1.0, 5.0, inf],
            [1.0, 0.0, 1.0, inf],
            [5.0, 1.0, 0.0, 2.0],
            [inf, inf, 2.0, 0.0],
        ]
    )


def test_reduce_distances_everywhere_by_default() -> None:
    completed = reduce_distances_by_shortest_paths(path_distances())
    # The direct 0 - 2 distance is replaced by the shorter path through 1.
    check.equal(completed[0, 2], 2.0)
    check.equal(completed[0, 3], 4.0)
    check.equal(completed[1, 3], 3.0)


def test_reduce_distances_infinite_only() -> None:
    completed = reduce_distances_by_shortest_paths(path_distances(), infinite_only=True)
    # Existing distances are kept, only infinite ones are shortest paths.
    check.equal(completed[0, 2], 5.0)
    check.equal(completed[0, 1], 1.0)
    check.equal(completed[2, 3], 2.0)
    check.equal(completed[0, 3], 4.0)
    check.equal(completed[1, 3], 3.0)
    check.is_true(np.array_equal(completed, completed.T))


def test_reduce_distances_infinite_only_disconnected() -> None:
    distances = np.array(
        [
            [0.0, 1.0, np.inf],
            [1.0, 0.0, np.inf],
            [np.inf, np.inf, 0.0],
        ]
    )
    completed = reduce_distances_by_shortest_paths(distances, infinite_only=True)
    # Disconnected pairs are set to twice the largest finite distance.
    check.equal(completed[0, 1], 1.0)
    check.equal(completed[0, 2], 2.0)


def test_embed_mds_infinite_only() -> None:
    qubo = np.zeros((4, 4))
    qubo[0, 1] = qubo[1, 0] = 1.0
    qubo[1, 2] = qubo[2, 1] = 0.5
    qubo[2, 3] = qubo[3, 2] = 2.0
    positions = embed_mds(qubo, shortest_path_infinite_only=True)
    check.equal(positions.shape, (4, 2))
    check.is_true(np.isfinite(positions).all())


def test_embed_mds_does_not_modify_input() -> None:
    qubo = np.array([[-1.0, 1.0, 2.0], [1.0, -2.0, 3.0], [2.0, 3.0, -3.0]])
    original = qubo.copy()
    embed_mds(qubo)
    check.is_true(np.array_equal(qubo, original))
