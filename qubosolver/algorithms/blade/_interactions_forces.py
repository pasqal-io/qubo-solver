from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from ._force import Force
from ._helpers import normalized_best_dist, normalized_interaction

logger = logging.getLogger(__name__)


def compute_target_weights_by_dist_limit(
    *,
    distance_matrix: np.ndarray,
    target_weights: np.ndarray,
    max_distance_to_walk: float,
) -> Any:
    target_distances = np.vectorize(normalized_best_dist, signature="(m,n)->(m,n)")(target_weights)
    np.fill_diagonal(target_distances, 0)
    distances_to_walk = (distance_matrix - target_distances) / 2
    np.fill_diagonal(distances_to_walk, 0)

    modulated_distances_to_walk = np.minimum(
        max_distance_to_walk, np.maximum(-max_distance_to_walk, distances_to_walk)
    )
    modulated_target_distances = distance_matrix - modulated_distances_to_walk * 2

    # the following lines rectify possible numerical errors
    rectified_modulated_target_distances = np.where(
        modulated_distances_to_walk == 0,
        distance_matrix,
        np.where(
            modulated_distances_to_walk > 0,
            np.maximum(modulated_target_distances, target_distances),
            np.minimum(modulated_target_distances, target_distances),
        ),
    )
    modulated_target_weights = np.vectorize(normalized_interaction, signature="(m,n)->(m,n)")(
        rectified_modulated_target_distances
    )

    assert not np.any(np.triu(np.isinf(modulated_target_weights), k=1))

    return modulated_target_weights


def compute_target_weights_distances_by_weight_diff_limit(
    *,
    n: int,
    distance_matrix: np.ndarray,
    unitary_vectors: np.ndarray,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    weight_relative_threshold: float,
) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        weight_differences = target_weights - current_weights
    weight_differences[range(n), range(n)] = 0
    logger.debug(f"{weight_differences=}")
    # significant_weight_difference = np.max(np.abs(weight_differences)) / 100

    weight_difference_threshold = np.max(np.abs(weight_differences)) * weight_relative_threshold
    logger.debug(f"{weight_difference_threshold=}")
    weight_differences[np.abs(weight_differences) < weight_difference_threshold] = (
        0.0  # or use exponentially decreasing value
    )
    logger.debug(f"new {weight_differences=}")

    step_target_weights = current_weights + weight_differences * (1 - weight_relative_threshold)
    logger.debug(f"{step_target_weights=}")
    step_target_distances = np.vectorize(normalized_best_dist, signature="(m,n)->(m,n)")(
        step_target_weights
    )
    logger.debug(f"{step_target_distances=}")

    distances_to_walk = (
        distance_matrix - step_target_distances
    ) / 2  # division by 2 because both forces will be applied on both atoms of each pair
    logger.debug(f"{distances_to_walk=}")

    weighted_vectors = weight_differences[:, :, np.newaxis] * unitary_vectors
    assert not np.any(np.isnan(weighted_vectors))
    logger.debug(f"{weighted_vectors=}")

    return weighted_vectors, distances_to_walk


def compute_interaction_forces(
    *,
    distance_matrix: np.ndarray,
    unitary_vectors: np.ndarray,
    qubo_graph: nx.Graph,
    weight_relative_threshold: float,
    max_distance_to_walk: float,
) -> Force:
    n = nx.number_of_nodes(qubo_graph)

    current_weights = np.vectorize(normalized_interaction, signature="(m,n)->(m,n)")(
        distance_matrix
    )
    logger.debug(f"{current_weights=}")
    target_weights = np.array(
        nx.adjacency_matrix(qubo_graph, nodelist=list(range(n)), weight="weight").toarray()
    )
    logger.debug(f"{target_weights=}")

    modulated_target_weights = compute_target_weights_by_dist_limit(
        distance_matrix=distance_matrix,
        target_weights=target_weights,
        max_distance_to_walk=max_distance_to_walk,
    )

    weighted_vectors, distances_to_walk = compute_target_weights_distances_by_weight_diff_limit(
        n=n,
        distance_matrix=distance_matrix,
        unitary_vectors=unitary_vectors,
        current_weights=current_weights,
        target_weights=modulated_target_weights,
        weight_relative_threshold=weight_relative_threshold,
    )

    return Force(weighted_vectors=weighted_vectors, distances_to_walk=np.abs(distances_to_walk))
