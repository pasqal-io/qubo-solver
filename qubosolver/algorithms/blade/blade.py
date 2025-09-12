from __future__ import annotations

import logging
from typing import Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from pulser.devices._device_datacls import BaseDevice
from sklearn.decomposition import PCA
import scipy

from ._dimension_shrinker import DimensionShrinker
from ._dist_constraints_forces import (
    compute_max_dist_constraint_forces,
    compute_min_dist_constraint_forces,
)
from ._distances_constraints_calculator import DistancesContraintsCalculator
from ._helpers import find_center, distance_matrix_from_positions
from ._interactions_forces import compute_interaction_forces
from ._qubo_mapper import Qubo
from .drawing import draw_graph_including_actual_weights

logger = logging.getLogger(__name__)


def update_positions(
    *,
    positions: np.ndarray,
    qubo_graph: nx.Graph,
    draw_step: bool = False,
    device: BaseDevice,
    weight_relative_threshold: float = 0.0,
    min_dist: float | None = None,
    max_dist: float | None = None,
    max_distance_to_walk: float | tuple[float, float, float] = np.inf,
) -> np.ndarray:
    """
    Compute vector moves and apply them on the positions of the nodes to make
    their interactions closer to the target QUBO.

    positions: Starting positions of the nodes.
    qubo_graph: Desired QUBO.
    draw_steps: Whether to draw the nodes and the forces.
    device: Used for its interaction coefficient.
    weight_relative_threshold: It is used to compute a weight difference
        threshold defining which weights differences are significant and should
        be considered. For this purpose, it is multiplied by the higher weight difference.
        It is also used to reduce the precision when targetting
        the objective weights.
    min_dist: If set, defined the minimum distance that should be met, and
        creates forces to enforce the constraint.
    max_dist: If set, defined the maximum distance that should be met, and
        creates forces to enforce the constraint.
    max_distance_to_walk: It set, limits the distance that nodes can walk
        when the forces are applied. It impacts the priorities
        of the forces because they only consider the slope of the differences
        in weights that can be targetting with this ceiling.
    """

    n = nx.number_of_nodes(qubo_graph)
    positions = np.array(positions, dtype=float)
    nb_positions, space_dimension = positions.shape

    if isinstance(max_distance_to_walk, tuple):
        max_distance_to_walk, min_constr_max_distance_to_walk, max_constr_max_distance_to_walk = (
            max_distance_to_walk
        )
    else:
        min_constr_max_distance_to_walk = np.inf
        max_constr_max_distance_to_walk = np.inf

    assert nb_positions == n

    position_differences = positions[np.newaxis, :] - positions[:, np.newaxis]
    distance_matrix = np.linalg.norm(position_differences, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        unitary_vectors = position_differences / distance_matrix[:, :, np.newaxis]
    unitary_vectors[range(n), range(n)] = np.zeros(space_dimension)
    logger.debug(f"{unitary_vectors=}")

    interaction_force = compute_interaction_forces(
        distance_matrix=distance_matrix,
        unitary_vectors=unitary_vectors,
        device=device,
        qubo_graph=qubo_graph,
        weight_relative_threshold=weight_relative_threshold,
        max_distance_to_walk=max_distance_to_walk,
    )
    min_constr_force = compute_min_dist_constraint_forces(
        min_dist=min_dist,
        distance_matrix=distance_matrix,
        unitary_vectors=unitary_vectors,
    )
    max_constr_force = compute_max_dist_constraint_forces(
        positions=positions,
        max_dist=max_dist,
    )

    interaction_resulting_forces = interaction_force.get_resulting_forces(
        interaction_force.get_temperature()
    )

    min_constr_resulting_forces = min_constr_force.get_resulting_forces(
        min_constr_force.get_temperature()
    )
    limited_min_constr_resulting_forces = (
        min_constr_resulting_forces
        * np.minimum(
            1,
            min_constr_max_distance_to_walk / np.linalg.norm(min_constr_resulting_forces, axis=-1),
        )[:, np.newaxis]
    )
    limited_min_constr_resulting_forces[min_constr_resulting_forces == 0] = 0

    max_constr_resulting_forces = max_constr_force.get_resulting_forces(
        max_constr_force.get_temperature()
    )
    limited_max_constr_resulting_forces = (
        max_constr_resulting_forces
        * np.minimum(
            1,
            max_constr_max_distance_to_walk / np.linalg.norm(max_constr_resulting_forces, axis=-1),
        )[:, np.newaxis]
    )
    limited_max_constr_resulting_forces[max_constr_resulting_forces == 0] = 0

    resulting_forces_vectors = (
        interaction_resulting_forces
        + limited_min_constr_resulting_forces
        + limited_max_constr_resulting_forces
    )
    logger.debug(f"{resulting_forces_vectors=}")

    assert not np.any(np.isinf(interaction_resulting_forces)) and not np.any(
        np.isnan(interaction_resulting_forces)
    )
    assert not np.any(np.isinf(min_constr_resulting_forces)) and not np.any(
        np.isnan(min_constr_resulting_forces)
    )
    assert not np.any(np.isinf(max_constr_resulting_forces)) and not np.any(np.isnan(positions))

    if draw_step:

        def keep_2_dims(a: np.ndarray) -> np.ndarray:
            return a[0:2]

        logger.debug(
            "Step that will be applied" + (" keeping only 2 dims" if positions.shape[1] > 2 else "")
        )
        plt.scatter(positions[:, 0], positions[:, 1])

        base_arrow_width = np.max(scipy.spatial.distance.pdist(positions)) / 100

        for u, force in enumerate(interaction_resulting_forces):
            if np.any(force):
                plt.arrow(
                    *keep_2_dims(positions[u]),
                    *keep_2_dims(force),
                    color="blue",
                    width=base_arrow_width,
                )

        for u, force in enumerate(min_constr_resulting_forces):
            if np.any(force):
                plt.arrow(
                    *keep_2_dims(positions[u]),
                    *keep_2_dims(force),
                    color="green",
                    width=base_arrow_width,
                )

        for u, force in enumerate(max_constr_resulting_forces):
            if np.any(force):
                plt.arrow(
                    *keep_2_dims(positions[u]),
                    *keep_2_dims(force),
                    color="black",
                    width=base_arrow_width,
                )

        for position, force in zip(positions, resulting_forces_vectors):
            plt.arrow(
                *keep_2_dims(position),
                *keep_2_dims(force),
                color="red",
                width=base_arrow_width * 0.4,
            )

        plt.gca().set_aspect("equal", "box")

        if max_dist is not None:
            center = find_center(positions)
            circle = plt.Circle(center, max_dist / 2, color="r", fill=False, clip_on=True)
            ax = plt.gca()
            ax.add_patch(circle)
        plt.show()

    for u, force in enumerate(resulting_forces_vectors):
        positions[u] += force

    assert not np.any(np.isnan(positions))

    if draw_step:
        logger.debug(f"Resulting positions = {dict(enumerate(positions))}")
        print(f"Current number of dimensions is {positions.shape[-1]}")
        print(
            f"{min_dist=}, {max_dist=}, current min dist = {np.min(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])}, current max dist = {np.max(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])}"
        )
        draw_graph_including_actual_weights(
            qubo_graph=qubo_graph, positions=positions, device=device
        )

    return positions


def evolve_with_forces_through_dim_change(
    *,
    qubo_graph: nx.Graph,
    device: BaseDevice,
    draw_steps: bool = False,
    starting_dimensions: int,
    final_dimensions: int,
    nb_steps: int,
    positions: np.ndarray,
    starting_min: float | None = None,
    start_ratio: float | None = None,
    final_ratio: float | None = None,
    compute_weight_relative_threshold_by_step: Callable[[int], float],
    compute_max_distance_to_walk_by_step: Callable[
        [int, float | None], float | tuple[float, float, float]
    ],
) -> tuple[np.ndarray, float | None]:
    dim_shrinker = DimensionShrinker(
        dimensions_to_remove=starting_dimensions - final_dimensions, steps=nb_steps
    )
    dist_constr_calc = DistancesContraintsCalculator(
        target_qubo=Qubo.from_graph(qubo_graph).as_matrix(),
        device=device,
        starting_min=starting_min,
        starting_ratio=start_ratio,
        final_ratio=final_ratio,
    )
    assert np.unique(positions, axis=0).shape == positions.shape

    for step in range(0, nb_steps):
        draw_step = draw_steps is True or (isinstance(draw_steps, list) and step in draw_steps)

        if draw_step:
            print(f"{step=}")
        scaling, min_dist, max_dist = dist_constr_calc.compute_scaling_min_max(
            positions=positions,
            step_cursor=(step + 1) / nb_steps,
            plot=draw_step,
        )
        assert np.unique(positions, axis=0).shape == positions.shape
        positions = scaling * positions
        if draw_step:
            print(
                f"After {scaling=}, max/min is {np.max(scipy.spatial.distance.pdist(positions))}/{np.min(scipy.spatial.distance.pdist(positions))} with target {max_dist}/{min_dist}"
            )
        assert not np.any(np.isinf(positions)) and not np.any(np.isnan(positions))

        positions = update_positions(
            positions=positions,
            qubo_graph=qubo_graph,
            draw_step=draw_step,
            device=device,
            weight_relative_threshold=compute_weight_relative_threshold_by_step(step),
            min_dist=min_dist,
            max_dist=max_dist,
            max_distance_to_walk=compute_max_distance_to_walk_by_step(step, max_dist),
        )
        assert np.unique(positions, axis=0).shape == positions.shape
        assert not np.any(np.isinf(positions)) and not np.any(np.isnan(positions))
        positions = dim_shrinker.applied_step(positions)

    removed_position_dims = positions[:, final_dimensions:]
    assert np.all(
        np.isclose(removed_position_dims, 0)
    ), f"Shrinked dimensions {removed_position_dims=} should only contain zeros"

    return positions[:, :final_dimensions], min_dist


def generate_random_positions(qubo: np.ndarray, device: BaseDevice, dimension: int) -> np.ndarray:
    return np.random.uniform(size=(len(qubo), dimension))


def evolve_with_dimension_transition(
    qubo: np.ndarray,
    *,
    device: BaseDevice,
    draw_steps: bool | list[int],
    dimensions: list[int],
    starting_min: float | None,
    pca: bool,
    steps_per_round: int,
    compute_weight_relative_threshold: Callable[[float], float],
    compute_max_distance_to_walk: Callable[
        [float, float | None], float | tuple[float, float, float]
    ],
    qubo_graph: nx.Graph,
    positions: np.ndarray,
    final_ratio: float,
    total_steps: int,
    dim_idx: int,
    start_ratio: float,
) -> tuple[np.ndarray, float | None]:

    starting_dimensions = dimensions[dim_idx]
    final_dimensions = dimensions[dim_idx + 1]
    performed_steps = dim_idx * steps_per_round
    target_performed_steps = (dim_idx + 1) * steps_per_round

    curr_starting_step_idx = dim_idx * steps_per_round

    def compute_weight_relative_threshold_by_step(steps: int) -> float:
        progress = (min(performed_steps, target_performed_steps) + steps) / total_steps
        return compute_weight_relative_threshold(progress)

    def compute_max_distance_to_walk_by_step(
        steps: int, max_radial_dist: float | None
    ) -> float | tuple[float, float, float]:
        progress = (min(performed_steps, target_performed_steps) + steps) / total_steps
        return compute_max_distance_to_walk(progress, max_radial_dist)

    if final_dimensions < starting_dimensions and pca:
        pca_inst = PCA(n_components=starting_dimensions)
        positions = pca_inst.fit_transform(positions)

    if final_dimensions > starting_dimensions:
        variance = np.mean(np.var(positions, axis=0))
        noise_factor = variance / 20
        positions_noise = np.random.uniform(
            low=-noise_factor,
            high=noise_factor,
            size=(len(qubo), final_dimensions - starting_dimensions),
        )
        positions = np.concatenate((positions, positions_noise), axis=1)
        starting_dimensions = final_dimensions

    positions, starting_min = evolve_with_forces_through_dim_change(
        qubo_graph=qubo_graph,
        device=device,
        draw_steps=(
            draw_steps
            if isinstance(draw_steps, bool)
            else (np.array(draw_steps) - curr_starting_step_idx).tolist()
        ),
        starting_dimensions=starting_dimensions,
        final_dimensions=final_dimensions,
        positions=positions,
        nb_steps=steps_per_round,
        starting_min=starting_min,
        start_ratio=start_ratio,
        final_ratio=final_ratio,
        compute_weight_relative_threshold_by_step=compute_weight_relative_threshold_by_step,
        compute_max_distance_to_walk_by_step=compute_max_distance_to_walk_by_step,
    )

    return positions, starting_min


def em_blade(
    qubo: np.ndarray,
    *,
    device: BaseDevice,
    draw_steps: bool | list[int] = False,
    dimensions: list[int] = [5, 4, 3, 2, 2, 2],
    starting_positions: Optional[np.ndarray] = None,
    pca: bool = False,
    steps_per_round: int = 200,
    enforce_min_max_dist_ratio: bool = False,
    compute_weight_relative_threshold: Callable[[float], float] = (lambda _: 0.1),
    compute_max_distance_to_walk: Callable[
        [float, float | None], float | tuple[float, float, float]
    ] = (lambda x, max_radial_dist: np.inf),
    starting_ratio_factor: int = 2,
) -> np.ndarray:
    """
    Embed a problem using the BLaDE algorithm.

    Compute positions for nodes so that their interactions according to a device
    approach the desired values at best. The result can be used as an embedding.
    Its prior target is on interaction matrices or QUBOs, but it can also be used
    for MIS with limitations if the adjacency matrix is converted into a QUBO.
    The general principle is based on the Fruchterman-Reingold algorithm.

    qubo: QUBO matrix
    device: Used for its interaction coefficient, and for its minimum and
        maximum distances if `enforce_min_max_dist_ratio` is enabled.
    draw_steps: Whether to draw the nodes and the forces.
    dimensions: List of numbers of dimensions to explore one
        after the other. A list with one value is equivalent to a list containing
        twice the same value. For a 2D embedding, the last value should be 2.
        Increasing the number of intermediate dimensions can help to escape
        from local minima.
    starting_positions: If provided, initial positions to start from. Otherwise,
        random positions will be generated.
    pca: Whether to apply Principal Component Analysis to prioritize dimensions
        to keep when transitioning from a space to a space with fewer dimensions.
        It is disabled by default because it can raise an error when there are
        too many dimensions compared to the number of nodes.
    steps_per_round: Number of elementary steps to perform for each dimension
        transition, where at each step move vectors are computed and applied
        on the nodes.
    enforce_min_max_dist_ratio: Whether to enforce the ratio between the
        maximal radial distance and the minimum pairwise distance. It does not
        directly enforces these distances but only the ratio. To meet the
        distance, you can rescale the positions and infer how it should impact
        the Rabi frequency. This may change in the future to directly meet
        the distances constraints and compute the Rabi frequency for you.
    compute_weight_relative_threshold: Function that is called at each step.
        It takes a float number between 0 and 1 that represents the progress
        on the steps. It must return a float number between 0 and 1 that gives
        a threshold determining which weights are significant (see
        `update_positions` to learn more).
    compute_max_distance_to_walk: Function that is called at each step.
        It takes a float number between 0 and 1 that represents the progress
        on the steps, and takes another argument that is set to `None` when
        `enforce_min_max_dist_ratio` is not enabled, otherwise, it is set to
        the maximum radial distance for the current step.
        It must return a float number that limits the distances
        nodes can move at one step  (see
        `update_positions` to learn more).
    starting_ratio_factor: When `enforce_min_max_dist_ratio` is enabled,
        defines a multiplying factor on the target ratio to start the evolution
        on a larger ratio, to let more flexibility in the beginning.
    """

    if len(dimensions) == 1:
        dimensions = [dimensions[0], dimensions[0]]

    assert len(dimensions) >= 2

    if isinstance(qubo, np.ndarray):
        assert not np.all(qubo == 0)
    else:
        assert not torch.all(qubo == 0)

    qubo_obj = Qubo.from_matrix(qubo)
    qubo_graph = qubo_obj.as_graph()

    if starting_positions is None:
        positions = generate_random_positions(qubo=qubo, device=device, dimension=dimensions[0])
    else:
        positions = starting_positions

    for u, v in nx.non_edges(qubo_graph):
        qubo_graph.add_edge(u, v, weight=0)

    if enforce_min_max_dist_ratio:
        if device.max_radial_distance is None:
            raise ValueError(
                "Cannot enforce the min max distance ratio on a device that has no maximal radial distance."
            )
        final_ratio = device.max_radial_distance / device.min_atom_distance
        starting_ratio = starting_ratio_factor * final_ratio
        steps_ratios = np.linspace(starting_ratio, final_ratio, len(dimensions))
    else:
        steps_ratios = [None] * len(dimensions)

    total_steps = steps_per_round * (len(dimensions) - 1)

    assert len(range(len(dimensions))) == len(steps_ratios)
    distance_matrix = distance_matrix_from_positions(positions)
    upper_diagonal_mask = np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)
    starting_min = np.min(distance_matrix[upper_diagonal_mask])

    for dim_idx, start_ratio, final_ratio in zip(
        range(len(dimensions) - 1), steps_ratios[:-1], steps_ratios[1:]
    ):
        positions, starting_min = evolve_with_dimension_transition(
            qubo=qubo,
            device=device,
            draw_steps=draw_steps,
            dimensions=dimensions,
            starting_min=starting_min,
            pca=pca,
            steps_per_round=steps_per_round,
            compute_weight_relative_threshold=compute_weight_relative_threshold,
            compute_max_distance_to_walk=compute_max_distance_to_walk,
            qubo_graph=qubo_graph,
            positions=positions,
            final_ratio=final_ratio,
            total_steps=total_steps,
            dim_idx=dim_idx,
            start_ratio=start_ratio,
        )

    return positions
