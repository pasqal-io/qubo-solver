# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Geometric search placing a cluster of QUBO variables, used by the partial greedy embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize
from shapely.geometry import MultiPolygon, Point, Polygon

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Find a better way to compute this ?
# Choose better than 50 ?
def _clamp_max_radial_distance(max_radial_distance: float) -> float:
    if np.isfinite(max_radial_distance).all():
        return max_radial_distance
    return 50.0


class VertexToPlace(TypedDict):
    """A variable still to place, with its relations to the other variables.

    Built by [`vertices_to_place`][]: `weight` is the variable's diagonal
    coefficient; `blocking_vertices`, `separated_vertices` and `neighbors_id`
    are the variables it blocks, is separated from, and interacts with, and
    `neighbors_weight` holds the QUBO coefficients to those neighbors.
    """

    id: int
    weight: np.ndarray
    blocking_vertices: np.ndarray
    separated_vertices: np.ndarray
    neighbors_id: np.ndarray
    neighbors_weight: np.ndarray


class WeightedZone:
    """A placed vertex together with its forbidden, blocking, and range zones.

    For a given placed vertex, identified by its id, this defines 3 zones or
    rings computed with shapely: forbidden, blocking, and the zone of edge
    values.

    Attributes:
        id (int): Identifier of vertex.
        x (float): x-coordinate.
        y (float): y-coordinate.
        weight (float): QUBO weight.
        forbidden_zone: Forbidden zone.
        blocking_zone: Blocking zone.
        range_zone: Zone of edge values.
        circle_end_forbidden: Circle delimiting end of forbidden_zone.
        circle_end_blocking: Circle delimiting end of blocking_zone.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        vertex_id: int,
        x: float,
        y: float,
        weight: float,
        min_distance: float,
        max_radial_distance: float = np.inf,
    ) -> None:
        """Initialize the zone geometry for a placed vertex.

        Args:
            vertex_id: Identifier of the vertex.
            x: x-coordinate of the vertex.
            y: y-coordinate of the vertex.
            weight: QUBO weight of the vertex, non-positive. A zero weight has an
                infinite blocking radius: the vertex blocks every other one.
            min_distance: Radius of the forbidden zone around the vertex.
            max_radial_distance: Radius of the search disc around the origin, which
                contains the vertex. It bounds the drawn blocking disc, as shapely
                cannot draw an infinite one.
        """
        self.id = vertex_id
        self.x = x
        self.y = y
        self.weight = weight

        self.radius1 = min_distance
        # Blocking radius: the interaction U(r) = 1 / r**6 exceeds the detuning -Q_ii / 2 (see
        # compute_distance_interaction_matrix). Keep in sync with its blocking threshold.
        self.radius2 = (-weight / 2) ** (-1 / 6) if weight else float("inf")

        self.radius3 = 1.5

        self.point = Point(self.x, self.y)
        self.forbidden_zone = self.point.buffer(self.radius1)
        # The vertex and the search disc lie within the search radius R of the origin, so a disc of
        # radius 3 R around the vertex covers the search disc, despite its polygonal approximation.
        search_radius = _clamp_max_radial_distance(max_radial_distance)
        self.circle_end_forbidden = self.point.buffer(min(self.radius2, 3 * search_radius))
        self.circle_end_blocking = self.point.buffer(self.radius3)
        self.blocking_zone = self.circle_end_forbidden.difference(self.forbidden_zone)
        self.range_zone = self.circle_end_blocking.difference(self.circle_end_forbidden)


def compute_min_max_distances(
    interactions: NDArray[np.float64], *, max_min_dist_ratio: float
) -> tuple[float, float]:
    """Compute the minimum inter-atom distance and the maximum radial distance.

    The minimum distance is derived from the strongest (closest-required)
    interaction in the whole matrix, consistent with the adimensional
    interaction law ``U(r) = 1 / r ** 6``.

    Args:
        interactions (Matrix): QUBO interaction matrix.
        max_min_dist_ratio (float): Maximum allowed ratio between the maximum
            radial distance and the minimum inter-atom distance.

    Returns:
        tuple[float, float]: The minimum inter-atom distance and the maximum
            radial distance. Returns ``(0, np.inf)`` when
            ``max_min_dist_ratio`` is infinite (no distance limits).
    """
    if max_min_dist_ratio == np.inf:
        return 0.0, np.inf
    min_distance = float(np.max(np.triu(interactions, k=1)) ** (-1 / 6))
    max_radial_distance = min_distance * max_min_dist_ratio
    return min_distance, max_radial_distance


def compute_distance_interaction_matrix(
    qubo_matrix: NDArray[np.float64],
    neglecting_inter_distance: float = 1.5,
    neglecting_max_coefficient: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the matrix of interaction distances (adimensional, ``U(r) = 1 / r ** 6``).

    Args:
        qubo_matrix (np.ndarray): Matrix of qubo coefficients.
        neglecting_inter_distance (float, optional): Default distance value
            for neglecting interactions.
        neglecting_max_coefficient: Qubo coefficient from which we consider an
            interaction is neglecting.

    Returns:
        np.ndarray: distance matrix (adimensional units),
            diagonal elements are taken from `qubo_matrix`,
            neglecting interactions (coefficients less than 1) are set to the
            `neglecting_inter_distance` value. The rest derived from ``U(r) = 1 / r ** 6``.
    """
    diag_mat = np.diag(qubo_matrix)
    # note: blocking interactions already set to 0
    dist_matrix = np.diag(diag_mat)

    # neglecting interactions
    dist_matrix[(qubo_matrix < neglecting_max_coefficient) & (qubo_matrix >= 0.0)] = (
        neglecting_inter_distance
    )

    # relevant interactions
    # A pair is blocking when setting both variables to 1 does not pay off. Under x^T Q x with
    # symmetric Q, that is Q_ii + Q_jj + 2 Q_ij > min(Q_ii, Q_jj), i.e. Q_ij > -max(Q_ii, Q_jj) / 2.
    # On the device (U_ij = Q_ij, detuning -Q_ii / 2, see `extract_qubo`), the interaction
    # exceeds the smaller detuning. Keep in sync with WeightedZone.radius2.
    cond = (
        (qubo_matrix > neglecting_max_coefficient)
        & (qubo_matrix < -diag_mat[:, None] / 2)
        & (qubo_matrix < -diag_mat[None, :] / 2)
    )

    if cond.any():
        # set value to the adimensional interaction distance
        # Rydberg blockade radius
        dist_matrix[cond] = qubo_matrix[cond].astype(np.float64) ** (-1 / 6)
    # set diagonal elements to `qubo_matrix` diagonal
    dist_matrix[range(len(dist_matrix)), range(len(dist_matrix))] = diag_mat
    return dist_matrix


def vertices_to_place(
    dist_matrix: NDArray[np.float64],
    qubo_matrix: NDArray[np.float64],
    separation_threshold: float = 1.5,
) -> dict[int, VertexToPlace]:
    """Obtain the dictionary of vertices to place, with blocking, separated and neighbors.

    I.e., qubo variables to still consider for solving subproblems.

    Blocking vertices are vertices that have interation distances of 0 for a current vertex.
    Separated ones are vertices that have interation distances equal to `separation_threshold`.
    Neighbors have the distances close to a vertex, or less than `separation_threshold`.

    Args:
        dist_matrix (np.ndarray): interaction distances computed from
            `compute_distance_interaction_matrix`.
        qubo_matrix (np.ndarray): Matrix of qubo coefficients.
        separation_threshold (float): threshold we consider 2 vertices separated. Should be equal
            to the `neglecting_inter_distance` parameter from `compute_distance_interaction_matrix`.

    Returns:
        dict[int, VertexToPlace] : dictionary of placed vertices, with keys
            being the variables indices to place and values the blocking, separated and neighbors.
    """
    vertices_dict: dict[int, VertexToPlace] = {}
    n = dist_matrix.shape[0]

    for i in range(n):
        distances = dist_matrix[i, :].ravel()
        # possible to check for 0 as `compute_distance_interaction_matrix` sets to 0 some elements
        # should be <= 0.1 ?
        blocking_vertices = np.argwhere(distances == 0)
        blocking_vertices = blocking_vertices[blocking_vertices != i]

        # should be >= separation_threshold
        separated_vertices = np.argwhere(distances == separation_threshold)
        separated_vertices = separated_vertices[separated_vertices != i]

        # 1e-2 is an arbitrary value to consider them neighbor.
        neighbors = np.argwhere((distances > 1e-2) & (distances < separation_threshold))  # noqa: PLR2004 (numerical tolerance)
        neighbors = neighbors[neighbors != i]

        vertices_dict[i] = {
            "id": i,
            "weight": dist_matrix[i, i],
            "blocking_vertices": blocking_vertices,
            "separated_vertices": separated_vertices,
            "neighbors_id": neighbors,
            "neighbors_weight": qubo_matrix[i, neighbors],
        }
    return vertices_dict


def zone_intersection(
    current_intersection_result: Polygon | MultiPolygon,
    list_vertices: np.ndarray,
    placed_vertices: dict[int, WeightedZone],
    attribute: str,
) -> Polygon | MultiPolygon:
    """Calculate intersection of zones for a given attribute of weighted zone.

    Args:
        current_intersection_result (Polygon): Current intersection geometry.
        list_vertices (np.ndarray): A list of vertices to loop over.
        placed_vertices (dict[int, WeightedZone]): Placed vertices to access zones.
        attribute (str): The attribute of the weighted zone for intersection.

    Returns:
        Polygon: The resulting geometry of intersection.
    """
    for vertex in list_vertices:
        if current_intersection_result.is_empty:
            break
        current_intersection_result = current_intersection_result.intersection(
            getattr(placed_vertices[vertex.item()], attribute)
        )
    return current_intersection_result


def separated_zone_intersection(
    current_intersection_result: Polygon | MultiPolygon,
    list_vertices: np.ndarray,
    placed_vertices: dict[int, WeightedZone],
) -> Polygon | MultiPolygon:
    """Calculate intersection of zones for separated vertices.

    Args:
        current_intersection_result (Polygon): Current intersection geometry.
        list_vertices (list): A list of vertices to loop over.
        placed_vertices (dict[int, WeightedZone]): Placed vertices to access zones.

    Returns:
        Polygon: The resulting geometry of intersection.
    """
    for vertex in list_vertices:
        if current_intersection_result.is_empty:
            break
        free_zone = current_intersection_result.difference(
            placed_vertices[vertex.item()].circle_end_blocking
        )
        current_intersection_result = current_intersection_result.intersection(free_zone)
    return current_intersection_result


def forbidden_zone_intersection(
    current_intersection_result: Polygon | MultiPolygon, placed_vertices: dict[int, WeightedZone]
) -> Polygon | MultiPolygon:
    """Calculate difference of zones with placed vertices.

    Args:
        current_intersection_result (Polygon): Current intersection geometry.
        placed_vertices (dict[int, WeightedZone]): Placed vertices to access zones.

    Returns:
        Polygon: The resulting geometry.
    """
    for zone in placed_vertices.values():
        if current_intersection_result.is_empty:
            break
        current_intersection_result = current_intersection_result.difference(zone.forbidden_zone)
    return current_intersection_result


def random_point_in_geometry(
    geometry: Polygon | MultiPolygon,
    max_trials: int = 10000,
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Obtain a random point from a geometry.

    Args:
        geometry (Polygon): 2D Geometry to sample from.
        max_trials (int, optional): Number of maximum trials. Defaults to 10000.
        rng: Random number generator for reproducibility.

    Returns:
        Coordinates of the sampled point.

    Raises:
        ValueError: If `max_trials` is not a positive integer.
    """
    if max_trials <= 0:
        msg = f"max_trials must be a positive integer, got {max_trials}."
        raise ValueError(msg)

    minx, miny, maxx, maxy = geometry.bounds
    p = np.zeros(2)

    for _ in range(max_trials):
        p[0] = minx + (maxx - minx) * rng.random()
        p[1] = miny + (maxy - miny) * rng.random()

        if geometry.contains(Point(p[0].item(), p[1].item())):
            break

    return p


def cost_interaction_point_continuous(  # noqa: PLR0913, PLR0917
    pos_new: npt.NDArray[np.float64],
    placed_points: list[list[float]],
    Q_target: list[float],
    blocked_indices: list[int],
    min_distance: float,
    max_radial_distance: float,
) -> float:
    """Cost for BFGS search.

    Args:
        pos_new (list[float]): Input position.
        placed_points (list[tuple[float, float]]): Placed points.
        Q_target (list[float]): Weights.
        blocked_indices (list): List of blocked indices.
        min_distance (float): Minimum allowed distance to any placed point.
        max_radial_distance (float): Maximum allowed radial distance from origin.

    Returns:
        float: Cost evaluation.
    """
    cost = 0.0
    for i, (xi, yi) in enumerate(placed_points):
        dx = pos_new[0] - xi
        dy = pos_new[1] - yi
        dist2 = dx**2 + dy**2

        interaction = 1 / (dist2**3) if dist2 != 0 else np.inf

        diff = Q_target[i] - interaction

        if i in blocked_indices:
            penalty = max(0.0, Q_target[i] - interaction)  # penalize only if interaction < Q
            cost += penalty**2
        else:
            cost += diff**2

    penalty_coeff = sum([coeff**2 for coeff in Q_target])
    if np.linalg.norm(pos_new) > max_radial_distance:
        cost += float(penalty_coeff * (1 + np.linalg.norm(pos_new) - max_radial_distance))

    for placed_point in placed_points:
        dist = np.linalg.norm(np.array(pos_new) - np.array(placed_point))
        if dist < min_distance:
            cost += float(penalty_coeff * (1 + min_distance - dist))

    return cost


def bfgs_placement(  # noqa: PLR0913, PLR0917
    vertex: int,
    init_guess: NDArray[np.float64],
    placed_vertices: dict[int, WeightedZone],
    matrix: NDArray[np.float64],
    dict_vertices_to_place: dict,
    min_distance: float,
    max_radial_distance: float,
) -> OptimizeResult:
    """BFGS search for placing vertices.

    Args:
        vertex (int): Vertex to place.
        init_guess (Vector): Initial guess for the BFGS search.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.
        matrix (np.ndarray): qubo matrix.
        dict_vertices_to_place (dict): Vertices to place.
        min_distance (float): Minimum allowed distance to any placed point.
        max_radial_distance (float): Maximum allowed radial distance from origin.

    Returns:
        OptimizeResult: Result of BFGS.
    """
    placed_points: list[list[float]] = []
    Q_target: list[float] = []
    current_blocked_edges: list[int] = []
    counter = 0
    for vertex_key, vertex_value in placed_vertices.items():
        placed_points.append([vertex_value.x, vertex_value.y])
        Q_target.append(matrix[vertex_key, vertex].item())

        if vertex_key in dict_vertices_to_place[vertex]["blocking_vertices"]:
            current_blocked_edges.append(counter)
        counter = counter + 1

    return minimize(
        cost_interaction_point_continuous,
        x0=init_guess.astype(np.float64),
        args=(placed_points, Q_target, current_blocked_edges, min_distance, max_radial_distance),
        method="BFGS",
    )


def check_limit_zone(final_point: Point, max_radial_distance: float) -> bool:
    """Check if the new embedded vertex is within the limit zone.

    Args:
        final_point (Point): New embedded vertex.
        max_radial_distance (float): Maximum allowed radial distance from origin.

    Returns:
        bool: Returns True if point is within limit zone.
    """
    center_poly = Point(0, 0)
    limit_zone = center_poly.buffer(_clamp_max_radial_distance(max_radial_distance))
    return bool(limit_zone.contains(final_point))


def check_prohibited_zones(placed_vertices: dict[int, WeightedZone], final_point: Point) -> bool:
    """Verify if new embedded vertex is not within the forbidden zone of placed vertices.

    Args:
        placed_vertices (dict[int, WeightedZone]): Placed vertices
        final_point (Point): New embedded vertex.

    Returns:
        bool: Returns True if point is not within the forbidden zone of placed vertices.
    """
    checker = True
    for zone in placed_vertices.values():
        if zone.forbidden_zone.contains(final_point):
            return False

        if final_point.buffer(0.51).contains(Point(zone.x, zone.y)):
            return False
    return checker


def test_placing_vertex(  # noqa: PLR0913, PLR0917
    vertex: int,
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: dict[int, WeightedZone],
    min_distance: float,
    max_radial_distance: float,
    matrix: NDArray[np.float64],
    cost_function_thresold: float,
    tested_vertices: list[int],
    queue: list[int],
    rng: np.random.Generator,
) -> None:
    """Test placing vertex.

    Args:
        vertex (int): Vertex to place.
        dict_vertices_to_place (dict[int, VertexToPlace]): Vertices to place.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.
        min_distance (float): Minimum allowed distance to any placed point.
        max_radial_distance (float): Maximum allowed radial distance from origin.
        matrix (np.ndarray): Qubo matrix.
        cost_function_thresold (float): Threshold for cost function.
        tested_vertices (list[int]): Already tested vertices.
        queue (list[int]): Current queue of vertices to place.
        rng: Random number generator for reproducibility.
    """
    placed_ids = list(placed_vertices.keys())
    neighbors = dict_vertices_to_place[vertex]["neighbors_id"][
        np.isin(dict_vertices_to_place[vertex]["neighbors_id"], placed_ids)
    ]
    blockings = dict_vertices_to_place[vertex]["blocking_vertices"][
        np.isin(dict_vertices_to_place[vertex]["blocking_vertices"], placed_ids)
    ]
    separated = dict_vertices_to_place[vertex]["separated_vertices"][
        np.isin(dict_vertices_to_place[vertex]["separated_vertices"], placed_ids)
    ]

    center_poly = Point(0, 0)
    final_intersection = center_poly.buffer(_clamp_max_radial_distance(max_radial_distance))

    final_intersection = zone_intersection(
        final_intersection, blockings, placed_vertices, "blocking_zone"
    )
    final_intersection = zone_intersection(
        final_intersection, neighbors, placed_vertices, "range_zone"
    )
    final_intersection = separated_zone_intersection(final_intersection, separated, placed_vertices)
    final_intersection = forbidden_zone_intersection(final_intersection, placed_vertices)

    if not final_intersection.is_empty:
        # find new point to place
        init_guess = random_point_in_geometry(final_intersection, rng=rng)
        result_descent = bfgs_placement(
            vertex,
            init_guess,
            placed_vertices,
            matrix,
            dict_vertices_to_place,
            min_distance=min_distance,
            max_radial_distance=max_radial_distance,
        )
        final_point = Point(result_descent.x[0], result_descent.x[1])

        if (
            result_descent.fun < cost_function_thresold
            and check_prohibited_zones(placed_vertices, final_point)
            and check_limit_zone(final_point, max_radial_distance=max_radial_distance)
        ):
            # admit point in placed_vertices
            temp_zone = WeightedZone(
                vertex_id=vertex,
                x=result_descent.x[0],
                y=result_descent.x[1],
                weight=dict_vertices_to_place[vertex]["weight"].item(),
                min_distance=min_distance,
                max_radial_distance=max_radial_distance,
            )
            placed_vertices[vertex] = temp_zone

            # update queue
            for j_tensor in dict_vertices_to_place[vertex]["blocking_vertices"]:
                j = j_tensor.item()
                if j not in placed_vertices and j not in tested_vertices and j not in queue:
                    queue.append(j)
            for j_tensor in dict_vertices_to_place[vertex]["neighbors_id"]:
                j = j_tensor.item()
                if j not in placed_vertices and j not in tested_vertices and j not in queue:
                    queue.append(j)

    else:
        tested_vertices.append(vertex)


def obtain_vertice_to_test(
    vertices_list: list[int],
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: dict[int, WeightedZone],
    rng: np.random.Generator,
) -> int:
    """Select a vertex to test placing.

    Args:
        vertices_list (list[int]): list of vertices to select from.
        dict_vertices_to_place (dict[int, VertexToPlace]): dictionary of vertices to place.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.
        rng: Random number generator for reproducibility.

    Returns:
        int: identifier of a vertex to test placing.
    """
    best_score = 0.0
    idx: int = int(rng.integers(0, len(vertices_list)))
    chosen_vertice = vertices_list[idx]

    for vertex in vertices_list:
        current_score = 0.0

        for i, neighbor in enumerate(dict_vertices_to_place[vertex]["neighbors_id"]):
            if neighbor.item() in placed_vertices:
                current_score = (
                    current_score + dict_vertices_to_place[vertex]["neighbors_weight"][i].item()
                )

        if current_score > best_score:
            best_score = current_score
            chosen_vertice = vertex

    best_score = 0.0

    for vertex in vertices_list:
        current_score = 0.0
        for blocking_vertex in dict_vertices_to_place[vertex]["blocking_vertices"]:
            if blocking_vertex.item() in placed_vertices:
                current_score = current_score + 1

        if current_score > best_score:
            best_score = current_score
            chosen_vertice = vertex

    return chosen_vertice


def geometric_search(  # noqa: PLR0913, PLR0917
    matrix: NDArray[np.float64],
    dict_vertices_to_place: dict[int, VertexToPlace],
    first_vertex: int,
    cost_function_thresold: float,
    min_distance: float,
    max_radial_distance: float,
    rng: np.random.Generator,
) -> dict[int, WeightedZone]:
    """Search an embeddable subproblem on the device for solving it during the decomposition.

    Args:
        matrix (np.ndarray): Full updated qubo matrix.
        dict_vertices_to_place (dict[int, VertexToPlace]): Vertices to place.
        first_vertex (int): First vertex to start search from.
        cost_function_thresold (float): Threshold between sum of target
            interactions and sum of interactions from embedding.
        min_distance (float): Minimum allowed distance between placed points.
        max_radial_distance (float): Maximum allowed radial distance from origin.
        rng (np.random.Generator): Random number generator used to break ties.

    Returns:
        dict[int, WeightedZone]: placed vertices (from the matrix variables)
            with coordinates representing embedded graph
    """
    # place first vertex
    zone0 = WeightedZone(
        vertex_id=first_vertex,
        x=0.0,
        y=0.0,
        weight=dict_vertices_to_place[first_vertex]["weight"].item(),
        min_distance=min_distance,
        max_radial_distance=max_radial_distance,
    )
    placed_vertices: dict[int, WeightedZone] = {}
    placed_vertices[first_vertex] = zone0

    # list of vertices with an edge with the first vertex
    add_block = np.concatenate(
        [
            dict_vertices_to_place[first_vertex]["blocking_vertices"],
            dict_vertices_to_place[first_vertex]["neighbors_id"],
        ]
    )

    tested_vertices: list[int] = []
    queue: list[int] = []
    for neighbor in add_block:
        test_placing_vertex(
            neighbor.item(),
            dict_vertices_to_place,
            placed_vertices,
            min_distance=min_distance,
            max_radial_distance=max_radial_distance,
            matrix=matrix,
            cost_function_thresold=cost_function_thresold,
            tested_vertices=tested_vertices,
            queue=queue,
            rng=rng,
        )

    while queue:
        i = obtain_vertice_to_test(queue, dict_vertices_to_place, placed_vertices, rng)
        queue.remove(i)
        test_placing_vertex(
            i,
            dict_vertices_to_place,
            placed_vertices,
            min_distance=min_distance,
            max_radial_distance=max_radial_distance,
            matrix=matrix,
            cost_function_thresold=cost_function_thresold,
            tested_vertices=tested_vertices,
            queue=queue,
            rng=rng,
        )

    return placed_vertices
