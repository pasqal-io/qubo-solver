from __future__ import annotations


import numpy as np

from ._force import Force


def compute_min_dist_constraint_forces(
    *,
    min_dist: float | None,
    distance_matrix: np.ndarray,
    unitary_vectors: np.ndarray,
) -> Force:
    min_distances_to_walk = (
        np.maximum(0, min_dist - distance_matrix) / 2
        if min_dist is not None
        else np.full_like(distance_matrix, 0)
    )
    min_weights = min_distances_to_walk
    min_weighted_vectors = -min_weights[:, :, np.newaxis] * unitary_vectors
    min_force = Force(
        weighted_vectors=min_weighted_vectors, distances_to_walk=min_distances_to_walk
    )

    return min_force


def compute_max_dist_constraint_forces(
    *,
    positions: np.ndarray,
    max_dist: float | None,
) -> Force:
    distances_from_center = np.linalg.norm(positions, axis=1)
    max_distances_to_walk = (
        np.maximum(0, distances_from_center - max_dist)
        if max_dist is not None
        else np.full_like(distances_from_center, 0)
    )
    max_weights = max_distances_to_walk
    max_unitary_vectors = positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]
    max_unitary_vectors[positions == 0] = 0
    max_weighted_vectors = -max_weights[:, np.newaxis] * max_unitary_vectors
    assert not np.any(np.isinf(max_weighted_vectors))
    max_force = Force(
        weighted_vectors=max_weighted_vectors, distances_to_walk=max_distances_to_walk
    )

    return max_force
