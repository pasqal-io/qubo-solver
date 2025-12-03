from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ._helpers import normalized_interaction, distance_matrix_from_positions


def compute_best_scaling_for_interactions(
    target_interactions: np.ndarray,
    embedded_interactions: np.ndarray,
    filter_differences: bool = True,
    plot: bool = False,
) -> Any:
    embedded_interactions_triu = embedded_interactions[
        np.triu_indices_from(embedded_interactions, k=1)
    ]
    target_interactions_triu = target_interactions[np.triu_indices_from(target_interactions, k=1)]

    differences = embedded_interactions_triu - target_interactions_triu

    percent = 100 - 2 / (len(target_interactions) - 1) * 10
    percentile = np.percentile(differences, percent)

    difference_ceiling = max(0.0, percentile)  # type: ignore
    limited_differences = np.minimum(differences, difference_ceiling)

    if plot:
        print(f"{percent=}, {percentile=}, {difference_ceiling=}, {max(limited_differences)=}")

        ax = sns.violinplot(
            {"differences": differences, "limited_differences": limited_differences}, inner=None
        )
        sns.stripplot(
            {"differences": differences, "limited_differences": limited_differences},
            edgecolor="black",
            linewidth=1,
            palette=["white"] * 1,
            ax=ax,
        )
        plt.show()

        ax = sns.violinplot({"limited_differences": limited_differences}, inner=None)
        sns.stripplot(
            {"limited_differences": limited_differences},
            edgecolor="black",
            linewidth=1,
            palette=["white"] * 1,
            ax=ax,
        )
        plt.show()

    if filter_differences:
        # when no visible differences, use the input value to avoid rounding issues
        filtered_embedded_interactions_triu = np.where(
            differences == limited_differences,
            embedded_interactions_triu,
            target_interactions_triu + limited_differences,
        )

    best_scaling = (
        np.sum(filtered_embedded_interactions_triu**2)
        / np.sum(filtered_embedded_interactions_triu * target_interactions_triu)
    ) ** (1 / 6)

    assert not np.isnan(best_scaling)
    assert not np.isinf(best_scaling)
    assert best_scaling > 0

    return best_scaling


def compute_best_scaling_for_pos(
    target_interactions: np.ndarray, positions: np.ndarray, plot: bool = False
) -> Any:
    distance_matrix = distance_matrix_from_positions(positions)

    current_weights = np.vectorize(normalized_interaction, signature="(m,n)->(m,n)")(
        dist=distance_matrix
    )
    current_weights = np.triu(current_weights, k=1)

    return compute_best_scaling_for_interactions(
        target_interactions=target_interactions, embedded_interactions=current_weights, plot=plot
    )


@dataclasses.dataclass
class DistancesContraintsCalculator:
    target_interactions: np.ndarray
    starting_min: float | None
    starting_ratio: float | None
    final_ratio: float | None = None
    current_min: float | None = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.final_ratio is not None:
            assert self.starting_min is not None
            self.current_min = self.starting_min
        else:
            self.starting_ratio = None
            self.current_min = None

    def compute_scaling_min_max(
        self, positions: np.ndarray, step_cursor: float, plot: bool = False
    ) -> tuple[float, Optional[float], Optional[float]]:
        """step_cursor is between 0 (start) and 1 (end)"""
        assert 0 <= step_cursor <= 1

        scaling_factor = compute_best_scaling_for_pos(
            target_interactions=self.target_interactions, positions=positions, plot=plot
        )

        if self.final_ratio is None:
            return scaling_factor, None, None

        assert self.starting_ratio is not None

        step_ratio = self.final_ratio + (1 - step_cursor) * (self.starting_ratio - self.final_ratio)

        scaled_min = self.current_min * scaling_factor
        self.current_min, current_max = scaled_min, scaled_min * step_ratio

        return scaling_factor, self.current_min, current_max
