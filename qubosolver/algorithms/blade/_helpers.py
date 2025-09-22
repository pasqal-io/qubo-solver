from __future__ import annotations

from typing import Any

import numpy as np
from pulser.devices._device_datacls import BaseDevice


def find_center(
    positions: np.ndarray,
) -> Any:
    return np.mean(positions, axis=0)


def interaction(device: BaseDevice, dist: float) -> float:
    return device.rabi_from_blockade(dist)  # type: ignore


def best_dist(device: BaseDevice, weight: float) -> float:
    return device.rydberg_blockade_radius(weight)  # type: ignore


def distance_matrix_from_positions(positions: np.ndarray) -> np.ndarray:
    position_differences = positions[np.newaxis, :] - positions[:, np.newaxis]
    return np.linalg.norm(position_differences, axis=2)


def interaction_matrix_from_distances(
    distance_matrix: np.ndarray, *, device: BaseDevice
) -> np.ndarray:
    current_weights = np.vectorize(interaction, excluded=["device"], signature="(m,n)->(m,n)")(
        device=device, dist=distance_matrix
    )
    return np.triu(current_weights, k=1)


def interaction_matrix_from_positions(positions: np.ndarray, *, device: BaseDevice) -> np.ndarray:
    return interaction_matrix_from_distances(
        distance_matrix=distance_matrix_from_positions(positions),
        device=device,
    )


def normalized_distance(target: np.ndarray, actual: np.ndarray) -> np.floating[Any]:
    return np.linalg.norm(target - actual) / np.linalg.norm(target)
