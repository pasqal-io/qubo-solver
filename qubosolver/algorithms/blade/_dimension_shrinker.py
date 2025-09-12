from __future__ import annotations

from typing import Any

import numpy as np


def shrink_dimensions(*, positions: np.ndarray, dimensions_lengths: np.ndarray) -> Any:
    _, space_dimension = positions.shape
    assert (space_dimension,) == dimensions_lengths.shape

    minima_per_dim = np.min(positions, axis=0)
    maxima_per_dim = np.max(positions, axis=0)
    current_lengths = maxima_per_dim - minima_per_dim
    lengths_to_remove = np.maximum(current_lengths - dimensions_lengths, 0)

    new_minima = minima_per_dim + lengths_to_remove / 2
    new_maxima = maxima_per_dim - lengths_to_remove / 2
    shrinked_positions = np.maximum(new_minima, np.minimum(new_maxima, positions))
    normalized_positions = shrinked_positions - np.median(shrinked_positions, axis=0)
    assert not np.any(np.isnan(normalized_positions))

    return normalized_positions


class DimensionShrinker:
    def __init__(self, dimensions_to_remove: int, steps: int):
        self._dimensions_to_remove = dimensions_to_remove
        self._next_lengths_steps: list = []
        self._steps = steps
        self._step = 0

    def applied_step(self, positions: np.ndarray) -> np.ndarray:
        _, nb_dimensions = positions.shape
        assert self._dimensions_to_remove < nb_dimensions

        if len(self._next_lengths_steps) == 0:
            for dim in range(nb_dimensions - self._dimensions_to_remove, nb_dimensions):
                curr_dim_values = positions[:, dim]
                curr_dim_length = np.max(curr_dim_values) - np.min(curr_dim_values)
                self._next_lengths_steps.append(
                    list(np.linspace(0, curr_dim_length, num=self._steps, endpoint=True))[::-1]
                )

        dims_lengths = np.max(positions, axis=0) - np.min(positions, axis=0)

        for dim, next_lengths in zip(
            range(nb_dimensions - self._dimensions_to_remove, nb_dimensions),
            self._next_lengths_steps,
        ):
            dims_lengths[dim] = next_lengths[self._step]

        self._step += 1

        positions = shrink_dimensions(positions=positions, dimensions_lengths=dims_lengths)

        if self._step >= self._steps:
            assert np.all(
                np.isclose(positions[:, nb_dimensions - self._dimensions_to_remove :], 0)
            ), f"{positions[:, nb_dimensions - self._dimensions_to_remove:]=}"

        assert not np.any(np.isnan(positions))

        return positions
