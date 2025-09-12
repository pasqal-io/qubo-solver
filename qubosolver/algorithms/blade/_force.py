from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Force:
    weighted_vectors: np.ndarray
    distances_to_walk: np.ndarray

    def __post_init__(self) -> None:
        assert self.weighted_vectors.shape[:-1] == self.distances_to_walk.shape
        assert not np.any(np.isnan(self.weighted_vectors))

    def get_nb_dims(self) -> int:
        return len(self.weighted_vectors.shape)

    def get_temperature(self) -> Any | int:
        vector_weights = np.linalg.norm(self.weighted_vectors, axis=self.get_nb_dims() - 1)
        logger.debug(f"{vector_weights=}")
        # remove warning output
        with np.errstate(divide="ignore", invalid="ignore"):
            maximum_temperatures = self.distances_to_walk / vector_weights
        maximum_temperatures[vector_weights == 0] = np.inf
        logger.debug(f"{maximum_temperatures=}")

        min_temperature = np.min(maximum_temperatures)

        # the next condition fixes the case where at some index, weighted_vectors is almost zero,
        # and squared in vector_weights it becomes 0 (but distances_to_walk is not zero so the force will be infinite)
        return min_temperature if min_temperature != np.inf else 0

    def get_forces(self, temperature: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            forces = temperature * self.weighted_vectors
        forces[self.weighted_vectors == 0] = 0
        return forces

    def get_resulting_forces(self, temperature: float) -> Any | np.ndarray:
        forces = self.get_forces(temperature)

        if self.get_nb_dims() == 3:
            return np.sum(forces, axis=1)
        else:
            return forces
