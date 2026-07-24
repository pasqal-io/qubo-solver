from __future__ import annotations

import math
import random
from typing import TypedDict, List

import numpy as np
import torch
from scipy.optimize import OptimizeResult, minimize
from shapely.geometry import Point, Polygon

VertexToPlace = TypedDict(
    "VertexToPlace",
    {
        "id": int,
        "weight": torch.Tensor,
        "blocking_vertices": torch.Tensor,
        "separated_vertices": torch.Tensor,
        "neighbors_id": torch.Tensor,
        "neighbors_weight": torch.Tensor,
    },
)


def compute_min_max_distances(
    interactions: torch.Tensor, *, max_min_dist_ratio: float
) -> tuple[float, float]:
    if max_min_dist_ratio == torch.inf:
        return 0, torch.inf
    min_distance = float(np.max(np.triu(interactions, k=1)) ** (-1 / 6))
    max_radial_distance = min_distance * max_min_dist_ratio
    return min_distance, max_radial_distance


class WeightedZone:
    """
    For a given placed vertex, identified by its id, we define 3 zones or rings
    calculated by shapely (forbidden, blocking, and the zone of edge values).

    Attributes :
        id (int) : identifier of vertex.
        x (float) : x-coordinate.
        y (float) : y-coordinate.
        weight (float) : qubo weight.
        forbidden_zone : forbidden zone.
        blocking_zone : blocking zone.
        range_zone : zone of edge values.
        circle_end_forbidden: circle delimiting end of forbidden_zone.
        circle_end_blocking: circle delimiting end of blocking_zone.
    """

    def __init__(self, id: int, x: float, y: float, weight: float, min_distance: float) -> None:

        self.id = id
        self.x = x
        self.y = y
        self.weight = weight

        self.radius1 = min_distance
        self.radius2 = (-weight) ** (-1 / 6)

        self.radius3 = 15.0

        self.point = Point(self.x, self.y)
        self.forbidden_zone = self.point.buffer(self.radius1)
        self.circle_end_forbidden = self.point.buffer(self.radius2)
        self.circle_end_blocking = self.point.buffer(self.radius3)
        self.blocking_zone = self.circle_end_forbidden.difference(self.forbidden_zone)
        self.range_zone = self.circle_end_blocking.difference(self.circle_end_forbidden)


def compute_distance_interaction_matrix(
    qubo_matrix: torch.Tensor,
    neglecting_inter_distance: float = 1.5,
    neglecting_max_coefficient: float = 1.0,
) -> torch.Tensor:
    """Compute the matrix of interaction distances.

    Args:
        qubo_matrix (torch.Tensor): Matrix of qubo coefficients.
        neglecting_inter_distance (float, optional): Default distance value
            for neglecting interactions.
        neglecting_max_coefficient: Qubo coefficient from which we consider an
            interaction is neglecting.

    Returns:
        torch.Tensor: distance matrix,
            diagonal elements are taken from `qubo_matrix`,
            neglecting interactions (coefficients less than 1) are set to the
            `neglecting_inter_distance` value. The rest are interactions.
    """

    diag_mat = torch.diag(qubo_matrix)
    # note: blocking interactions already set to 0
    dist_matrix = torch.diag(diag_mat)

    # neglecting interactions
    dist_matrix[(qubo_matrix < neglecting_max_coefficient) & (qubo_matrix >= 0.0)] = (
        neglecting_inter_distance
    )

    # relevant interactions
    cond = (
        (qubo_matrix > neglecting_max_coefficient)
        & (qubo_matrix < -diag_mat[:, None])
        & (qubo_matrix < -diag_mat[None, :])
    )

    def rydberg_blockade_radius(x: float) -> float:
        return float(x ** (-1 / 6))

    if cond.any():
        # set value to rydberg_blockade_radius
        dist_matrix[cond] = qubo_matrix[cond].apply_(rydberg_blockade_radius)
    # set diagonal elements to `qubo_matrix` diagonal
    dist_matrix[range(len(dist_matrix)), range(len(dist_matrix))] = diag_mat
    return dist_matrix


def vertices_to_place(
    dist_matrix: torch.Tensor,
    qubo_matrix: torch.Tensor,
    separation_threshold: float = 1.5,
) -> dict[int, VertexToPlace]:
    """Obtain the dictionary of vertices to place
    (i.e., qubo variables to still consider for solving subproblems)
    with blocking, separated and neighbors vertices.

    Blocking vertices are vertices that have interation distances of 0 for a current vertex.
    Separated ones are vertices that have interation distances equal to `separation_threshold`.
    Neighbors have the distances close to a vertex, or less than `separation_threshold`.

    Args:
        dist_matrix (torch.Tensor): interaction distances computed from
            `compute_distance_interaction_matrix`.
        qubo_matrix (torch.Tensor): Matrix of qubo coefficients.
        separation_threshold (float): threshold we consider 2 vertices separated. Should be equal
            to the `neglecting_inter_distance` parameter from `compute_distance_interaction_matrix`.

    Returns:
        dict[int, VertexToPlace] : dictionary of placed vertices, with keys
            being the variables indices to place and values the blocking, separated and neighbors.
    """
    vertices_dict: dict[int, VertexToPlace] = dict()
    n = dist_matrix.shape[0]

    for i in range(n):

        distances = dist_matrix[i, :].ravel()
        # possible to check for 0 as `compute_distance_interaction_matrix` sets to 0 some elements
        # should be <= 0.1 ?
        blocking_vertices = torch.argwhere(torch.eq(distances, 0))
        blocking_vertices = blocking_vertices[blocking_vertices != i]

        # equality because the dist matrix has setting on neglecting_inter_distance = 15
        separated_vertices = torch.argwhere(torch.eq(distances, separation_threshold))
        separated_vertices = separated_vertices[separated_vertices != i]

        # 1e-2 is an arbitrary value to consider them neighbor.
        neighbors = torch.argwhere((distances > 1e-2) & (distances < separation_threshold))
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


def update_vertex_info_from_placed(
    vertex: int,
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: torch.Tensor,
) -> None:
    """Update `dict_vertices_to_place` based on `placed_vertices` for a `vertex`.

    Args:
        vertex (int): Vertex to update info in dict_vertices_to_place.
        dict_vertices_to_place (dict[int, VertexToPlace]): Current dictionary of vertices to place.
        placed_vertices (torch.Tensor): Placed vertices indices.
    """

    neighbors_notplaced = ~torch.isin(
        dict_vertices_to_place[vertex]["neighbors_id"], placed_vertices
    )
    temp_neighborsid_not_placed = dict_vertices_to_place[vertex]["neighbors_id"][
        neighbors_notplaced
    ]
    temp_neighborsweight_not_placed = dict_vertices_to_place[vertex]["neighbors_weight"][
        neighbors_notplaced
    ]

    temp_cond = (temp_neighborsweight_not_placed > -dict_vertices_to_place[vertex]["weight"]) | (
        -torch.tensor(
            [dict_vertices_to_place[n.item()]["weight"] for n in temp_neighborsid_not_placed]
        )
        < temp_neighborsweight_not_placed
    )
    temp = dict_vertices_to_place[vertex]["neighbors_id"][neighbors_notplaced][temp_cond]

    dict_vertices_to_place[vertex]["neighbors_id"] = dict_vertices_to_place[vertex]["neighbors_id"][
        neighbors_notplaced
    ]
    dict_vertices_to_place[vertex]["neighbors_weight"] = dict_vertices_to_place[vertex][
        "neighbors_weight"
    ][neighbors_notplaced]

    blocking_cond_notplaced = ~torch.isin(
        dict_vertices_to_place[vertex]["blocking_vertices"], placed_vertices
    )
    dict_vertices_to_place[vertex]["blocking_vertices"] = torch.cat(
        [
            dict_vertices_to_place[vertex]["blocking_vertices"][blocking_cond_notplaced],
            temp,
        ]
    )

    separated_cond_notplaced = ~torch.isin(
        dict_vertices_to_place[vertex]["separated_vertices"], placed_vertices
    )
    dict_vertices_to_place[vertex]["separated_vertices"] = dict_vertices_to_place[vertex][
        "separated_vertices"
    ][separated_cond_notplaced]

    # should be <= and >= ?
    # should be ~temp ?
    neighbord_cond = (
        dict_vertices_to_place[vertex]["neighbors_weight"]
        < -dict_vertices_to_place[vertex]["weight"]
    ) & (
        -torch.tensor(
            [
                dict_vertices_to_place[n.item()]["weight"]
                for n in dict_vertices_to_place[vertex]["neighbors_id"]
            ]
        )
        > dict_vertices_to_place[vertex]["neighbors_weight"]
    )
    dict_vertices_to_place[vertex]["neighbors_id"] = dict_vertices_to_place[vertex]["neighbors_id"][
        neighbord_cond
    ]
    dict_vertices_to_place[vertex]["neighbors_weight"] = dict_vertices_to_place[vertex][
        "neighbors_weight"
    ][neighbord_cond]


def positive_vertices_update(
    dict_vertices_to_place: dict[int, VertexToPlace], global_solution: torch.Tensor
) -> List[int]:
    """Remove vertices whose weight became positive during decomposition.

    Args:
        dict_vertices_to_place (dict[int, VertexToPlace]): current dictionary of vertices to place.
        global_solution (torch.Tensor): global solution of the qubo.
    """
    positive_vertices = []
    for key, value in dict_vertices_to_place.items():
        if value["weight"].item() >= 0:
            positive_vertices.append(key)
            if global_solution[key] == -1:
                global_solution[key] = 0
            for vertex in dict_vertices_to_place:
                cond_not_key = dict_vertices_to_place[vertex]["neighbors_id"] != key
                dict_vertices_to_place[vertex]["neighbors_id"] = dict_vertices_to_place[vertex][
                    "neighbors_id"
                ][cond_not_key]
                dict_vertices_to_place[vertex]["neighbors_weight"] = dict_vertices_to_place[vertex][
                    "neighbors_weight"
                ][cond_not_key]

                dict_vertices_to_place[vertex]["blocking_vertices"] = dict_vertices_to_place[
                    vertex
                ]["blocking_vertices"][dict_vertices_to_place[vertex]["blocking_vertices"] != key]
                dict_vertices_to_place[vertex]["separated_vertices"] = dict_vertices_to_place[
                    vertex
                ]["separated_vertices"][dict_vertices_to_place[vertex]["separated_vertices"] != key]

    for key in positive_vertices:
        dict_vertices_to_place.pop(key, None)
    return positive_vertices


def transfer_edge_values(
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: dict,
    global_solution: torch.Tensor,
    matrix: torch.Tensor,
) -> None:
    """Transfer the values of cut vertices between the embedded subgraph
    and the leftover subgraph if the value in subgraph was fixed to 1,
    and update the dictionary of left vertices and global solution.

    Args:
        dict_vertices_to_place (dict[int, VertexToPlace]): current dictionary of vertices to place.
        placed_vertices (dict): placed vertices.
        global_solution (torch.Tensor): global solution of the qubo.
        matrix (torch.Tensor): current qubo matrix.
    """

    # add weight of cut vertices if subsolution variable was fixed to 1
    for key in placed_vertices:
        if global_solution[key] == 1:
            for key2 in dict_vertices_to_place.keys():
                if key2 != key:
                    dict_vertices_to_place[key2]["weight"] = (
                        dict_vertices_to_place[key2]["weight"] + matrix[key][key2]
                    )
                    matrix[key2][key2] += matrix[key][key2]

    # update placed vertices
    for key in placed_vertices:
        dict_vertices_to_place.pop(key, None)

    placed_vertices_tensor = torch.tensor(list(placed_vertices.keys()))
    # update neighbors, separated and blocked vertices from vertices still to place
    for vertex in dict_vertices_to_place:
        update_vertex_info_from_placed(vertex, dict_vertices_to_place, placed_vertices_tensor)


def zone_intersection(
    current_intersection_result: Polygon,
    list_vertices: torch.Tensor,
    placed_vertices: dict[int, WeightedZone],
    attribute: str,
) -> Polygon:
    """Calculate intersection of zones for a given attribute of weighted zone.

    Args:
        current_intersection_result (Polygon): Current intersection geometry.
        list_vertices (torch.Tensor): A list of vertices to loop over.
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
    current_intersection_result: Polygon,
    list_vertices: torch.Tensor,
    placed_vertices: dict[int, WeightedZone],
) -> Polygon:
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
    current_intersection_result: Polygon, placed_vertices: dict[int, WeightedZone]
) -> Polygon:
    """Calculate difference of zones with placed vertices.

    Args:
        current_intersection_result (Polygon): Current intersection geometry.
        placed_vertices (dict[int, WeightedZone]): Placed vertices to access zones.

    Returns:
        Polygon: The resulting geometry.
    """

    for vertex in placed_vertices:
        if current_intersection_result.is_empty:
            break
        current_intersection_result = current_intersection_result.difference(
            placed_vertices[vertex].forbidden_zone
        )
    return current_intersection_result


def random_point_in_geometry(geometry: Polygon, max_trials: int = 10000) -> list[float]:
    """Obtain a random point from a geometry.

    Args:
        geometry (Polygon): 2D Geometry to sample from.
        max_trials (int, optional): Number of maximum trials. Defaults to 10000.

    Returns:
        list[float]: Coordinates of the sampled point.
    """

    minx, miny, maxx, maxy = geometry.bounds

    for _ in range(max_trials):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)

        if geometry.contains(p):

            break
    return [x, y]


def cost_interaction_point_continuous(
    pos_new: list[float],
    placed_points: list[tuple[float, float]],
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
        min_distance (float): Minimum distance to set between the points.
        max_radial_distance (float): Maximum distance to set between each
            each point and the origin.

    Returns:
        float: Cost evaluation.
    """

    cost = 0.0

    for i, (xi, yi) in enumerate(placed_points):
        dx = pos_new[0] - xi
        dy = pos_new[1] - yi
        dist2 = dx**2 + dy**2

        interaction = 1 / (dist2**3) if dist2 != 0 else torch.inf

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


def bfgs_placement(
    vertex: int,
    final_intersection: Polygon,
    placed_vertices: dict[int, WeightedZone],
    matrix: torch.Tensor,
    dict_vertices_to_place: dict,
    min_distance: float,
    max_radial_distance: float,
) -> OptimizeResult:
    """BFGS search for placing vertices.

    Args:
        vertex (int): Vertex to place.
        final_intersection (Polygon): Intersection of zones.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.
        matrix (torch.Tensor): qubo matrix.
        dict_vertices_to_place (dict): Vertices to place.
        min_distance (float): Minimum distance to set between the points.
        max_radial_distance (float): Maximum distance to set between each
            each point and the origin.

    Returns:
        OptimizeResult: Result of BFGS.
    """
    placed_points: list[list[float]] = list()
    Q_target: list[float] = list()
    current_blocked_edges: list[int] = list()
    counter = 0
    for vertex_key, vertex_value in placed_vertices.items():

        placed_points.append([vertex_value.x, vertex_value.y])
        Q_target.append(matrix[vertex_key, vertex].item())

        if vertex_key in dict_vertices_to_place[vertex]["blocking_vertices"]:
            current_blocked_edges.append(counter)
        counter = counter + 1

    init_guess = random_point_in_geometry(final_intersection)

    return minimize(
        cost_interaction_point_continuous,
        x0=init_guess,
        args=(placed_points, Q_target, current_blocked_edges, min_distance, max_radial_distance),
        method="BFGS",
    )


def check_limit_zone(final_point: Point, max_radial_distance: float) -> bool:
    """Check if the new embedded vertex is within the limit zone.

    Args:
        final_point (Point): New embedded vertex.
        max_radial_distance (float): Maximum radial distance.

    Returns:
        bool: Returns True if point is within limit zone.
    """

    center_poly = Point(0, 0)
    limit_zone = center_poly.buffer(max_radial_distance)
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
    for key in placed_vertices.keys():
        if placed_vertices[key].forbidden_zone.contains(final_point):
            return False

        if final_point.buffer(0.51).contains(Point(placed_vertices[key].x, placed_vertices[key].y)):
            return False
    return checker


def test_placing_vertex(
    vertex: int,
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: dict[int, WeightedZone],
    min_distance: float,
    max_radial_distance: float,
    matrix: torch.Tensor,
    cost_function_thresold: float,
    tested_vertices: list[int],
    queue: list[int],
) -> None:
    """Test placing vertex.

    Args:
        vertex (int): Vertex to place.
        dict_vertices_to_place (dict[int, VertexToPlace]): Vertices to place.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.
        min_distance (float): Minimum distance between vertices.
        max_radial_distance (float): Maximum radial distance.
        matrix (torch.Tensor): Qubo matrix.
        cost_function_thresold (float): Threshold for cost function.
        tested_vertices (list[int]): Already tested vertices.
        queue (list[int]): Current queue of vertices to place.
    """
    places_vertices_tensor = torch.tensor(list(placed_vertices.keys()))
    neighbors = dict_vertices_to_place[vertex]["neighbors_id"][
        torch.isin(dict_vertices_to_place[vertex]["neighbors_id"], places_vertices_tensor)
    ]
    blockings = dict_vertices_to_place[vertex]["blocking_vertices"][
        torch.isin(dict_vertices_to_place[vertex]["blocking_vertices"], places_vertices_tensor)
    ]
    separated = dict_vertices_to_place[vertex]["separated_vertices"][
        torch.isin(dict_vertices_to_place[vertex]["separated_vertices"], places_vertices_tensor)
    ]

    center_poly = Point(0, 0)
    final_intersection = center_poly.buffer(max_radial_distance)

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
        result_descent = bfgs_placement(
            vertex,
            final_intersection,
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
                id=vertex,
                x=result_descent.x[0],
                y=result_descent.x[1],
                weight=dict_vertices_to_place[vertex]["weight"].item(),
                min_distance=min_distance,
            )
            placed_vertices[vertex] = temp_zone

            # update queue
            for j in dict_vertices_to_place[vertex]["blocking_vertices"]:
                j = j.item()
                if j not in placed_vertices and j not in tested_vertices and j not in queue:
                    queue.append(j)
            for j in dict_vertices_to_place[vertex]["neighbors_id"]:
                j = j.item()
                if j not in placed_vertices and j not in tested_vertices and j not in queue:

                    queue.append(j)

    else:
        tested_vertices.append(vertex)


def obtain_vertice_to_test(
    vertices_list: list[int],
    dict_vertices_to_place: dict[int, VertexToPlace],
    placed_vertices: dict[int, WeightedZone],
) -> int:
    """Select a vertex to test placing.

    Args:
        vertices_list (list[int]): list of vertices to select from.
        dict_vertices_to_place (dict[int, VertexToPlace]): dictionary of vertices to place.
        placed_vertices (dict[int, WeightedZone]): Placed vertices.

    Returns:
        int: identifier of a vertex to test placing.
    """

    best_score = 0.0
    chosen_vertice = random.choice(vertices_list)

    for vertex in vertices_list:
        current_score = 0.0

        for i, neighbor in enumerate(dict_vertices_to_place[vertex]["neighbors_id"]):
            neighbor
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


def geometric_search(
    matrix: torch.Tensor,
    dict_vertices_to_place: dict[int, VertexToPlace],
    first_vertex: int,
    cost_function_thresold: float,
    min_distance: float,
    max_radial_distance: float,
) -> dict[int, WeightedZone]:
    """Search an embeddable subproblem
        for solving it during the decomposition.

    Args:
        matrix (torch.Tensor): Full updated qubo matrix.
        dict_vertices_to_place (dict[int, VertexToPlace]): Vertices to place.
        first_vertex (int): First vertex to start search from.
        cost_function_thresold (float): Threshold between sum of target
            interactions and sum of interactions from embedding.
        min_distance (float): Minimum distance between vertices.
        max_radial_distance (float): Maximum radial distance.

    Returns:
        dict[int, WeightedZone]: placed vertices (from the matrix variables)
            with coordinates representing embedded graph
    """

    # place first vertex
    zone0 = WeightedZone(
        id=first_vertex,
        x=0,
        y=0,
        weight=dict_vertices_to_place[first_vertex]["weight"].item(),
        min_distance=min_distance,
    )
    placed_vertices: dict[int, WeightedZone] = dict()
    placed_vertices[first_vertex] = zone0

    # list of vertices with an edge with the first vertex
    add_block = torch.cat(
        [
            dict_vertices_to_place[first_vertex]["blocking_vertices"],
            dict_vertices_to_place[first_vertex]["neighbors_id"],
        ]
    )

    tested_vertices: list[int] = list()
    queue: list[int] = list()
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
        )

    while queue:
        i = obtain_vertice_to_test(queue, dict_vertices_to_place, placed_vertices)
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
        )

    return placed_vertices


def distance_plan(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the distance between two 2D points.

    Args:
        x1 (float): x-coordinate first point.
        y1 (float): y-coordinate first point.
        x2 (float): x-coordinate second point.
        y2 (float): y-coordinate second point.

    Returns:
        float: distance
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def interaction_matrix_from_placed(
    placed_vertices: dict[int, WeightedZone],
) -> tuple[torch.Tensor, dict[int, int]]:
    """
    Compute interaction matrix corresponding to embedded subgraph.

    Args:
        dict_vertices_to_place (dict[int, dict]): Vertices to place.
        placed_vertices (dict[int, WeightedZone]): Vertices already placed.

    Returns:
        tuple[torch.Tensor, dict]: Interacton matrix of embedded subgraph
            with a mapping of indices of placed vertices.
    """

    mat = torch.zeros((len(placed_vertices), len(placed_vertices)))

    sorted_placed_vertices = dict(sorted(placed_vertices.items()))
    map_index_vertices = {k[0]: i for i, k in enumerate(sorted_placed_vertices.items())}

    for key, value in sorted_placed_vertices.items():
        for key2, value2 in sorted_placed_vertices.items():
            if key2 != key:
                dist = distance_plan(
                    value.x,
                    value.y,
                    value2.x,
                    value2.y,
                )
                mat[map_index_vertices[key]][map_index_vertices[key2]] = 1 / dist**6
            else:
                mat[map_index_vertices[key]][map_index_vertices[key]] = value.weight

    return mat, map_index_vertices


def update_global_solution(
    global_solution: torch.Tensor, sub_solution: torch.Tensor, mapping: dict[int, int]
) -> None:
    """Transfer solution of the subgraph values to the global one.

    Args:
        global_solution (torch.Tensor): Current global solution.
        sub_solution (torch.Tensor): Current solution of subgraph.
        mapping (dict[int, int]): Mapping of indices.
    """
    for key, value in mapping.items():
        if global_solution[key] == -1:
            global_solution[key] = sub_solution[value]


def last_target_matrix(
    last_indices: list[int], matrix: torch.Tensor
) -> tuple[torch.Tensor, dict[int, int]]:
    """Obtain the last qubo matrix to solve
        after the decomposition algorithm.

    Args:
        last_indices (list[int]): List of last indices.
        matrix (torch.Tensor): Original qubo matrix.

    Returns:
        tuple[torch.Tensor, dict[int, int]]: Submatrix and mapping
            of indices from the last target qubo to the original
            qubo.
    """
    indices = sorted(last_indices)
    sub_matrix = matrix[np.ix_(indices, indices)]
    mapping = {index: i for i, index in enumerate(indices)}
    return sub_matrix, mapping
