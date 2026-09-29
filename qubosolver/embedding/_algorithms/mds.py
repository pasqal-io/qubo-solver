"""Multi-dimensional scaling (MDS) of a QUBO, used as a warm start for BLaDE."""

from __future__ import annotations

import numpy as np
from qoolqit.embedding.algorithms.blade._helpers import normalized_best_dist
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import smacof


def reduce_distances_by_shortest_paths(
    distances: np.ndarray, *, infinite_only: bool = False
) -> np.ndarray:
    """Return shortest-path distances.

    Args:
        distances: A square symmetric array of pairwise distances.
        infinite_only: If True, only infinite distances are replaced.

    Returns:
        A square symmetric array of finite pairwise distances. Disconnected pairs
        are set to twice the largest finite distance.
    """
    processed = shortest_path(distances, directed=False)

    if infinite_only:
        finite = ~np.isinf(distances)
        processed[finite] = distances[finite]

    infinite = np.isinf(processed)
    processed[infinite] = 2 * processed[~infinite].max()

    return np.asarray(processed)


def embed_mds(
    matrix: np.ndarray,
    max_iter: int = 3000,
    shortest_path_infinite_only: bool = False,
) -> np.ndarray:
    """Compute 2D positions for a QUBO with multi-dimensional scaling.

    Args:
        matrix: A square symmetric QUBO matrix with non-negative off-diagonal terms.
        max_iter: Maximum number of SMACOF iterations.
        shortest_path_infinite_only: If True, shortest paths only replace the distances
            of zero interactions. If False, they replace every distance.

    Returns:
        An array of shape ``(n, 2)`` with the position of each variable.
    """
    # The diagonal is irrelevant here, and a negative one would emit NaN warnings.
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    distances = normalized_best_dist(matrix, format="sym")
    distances = reduce_distances_by_shortest_paths(
        distances, infinite_only=shortest_path_infinite_only
    )
    coords, _ = smacof(distances, n_components=2, n_init=4, max_iter=max_iter, eps=1e-9)
    return np.asarray(coords)
