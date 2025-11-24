from __future__ import annotations

from typing import Any

import numpy as np


def normalized_interaction(dist: float) -> float:
    return 1 / dist**6


def normalized_best_dist(weight: float) -> float:
    return (1 / weight) ** (1 / 6)  # type: ignore[no-any-return]


def distance_matrix_from_positions(positions: np.ndarray) -> np.ndarray:
    position_differences = positions[np.newaxis, :] - positions[:, np.newaxis]
    return np.linalg.norm(position_differences, axis=2)


def interaction_matrix_from_distances(distance_matrix: np.ndarray) -> np.ndarray:
    current_weights = np.vectorize(normalized_interaction, signature="(m,n)->(m,n)")(
        distance_matrix
    )
    return np.triu(current_weights, k=1)


def interaction_matrix_from_positions(positions: np.ndarray) -> np.ndarray:
    return interaction_matrix_from_distances(
        distance_matrix=distance_matrix_from_positions(positions)
    )


def normalized_distance(target: np.ndarray, actual: np.ndarray) -> np.floating[Any]:
    return np.linalg.norm(target - actual) / np.linalg.norm(target)
