"""Internal building blocks for the QUBO decomposition algorithm.

The decomposition algorithm iteratively extracts small sub-problems from a
large QUBO instance, solves each on the quantum device, and stitches the
partial results back into a global solution.  This module provides the
stateful data structures and the three core steps of that loop:

1. `extract_subqubo` — select the next cluster of variables via
   geometric search and build the corresponding sub-matrix.
2. *Solve* — performed externally (quantum or classical).
3. `update` — merge the sub-solution into the global solution, propagate
   edge values to the remaining variables, and return a complete
   :class:`~qubosolver.types.Solution` once all variables are assigned.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import torch

from qubosolver import DecompositionConfig
from qubosolver import Solution, Matrix, matrix, Bitstring, bitstring, vectori, torch_rng
from qubosolver import Instance as QUBOInstanceBase
from ._algorithms.decompose import (
    compute_distance_interaction_matrix,
    compute_min_max_distances,
    geometric_search,
    interaction_matrix_from_placed,
    last_target_matrix,
    transfer_edge_values,
    update_global_solution,
    vertices_to_place,
    positive_vertices_update,
    VertexToPlace,
    WeightedZone,
)


@dataclass
class Config:
    """Internal configuration for the decomposition algorithm.

    Attributes:
        neglecting_inter_distance: Distance beyond which interactions are ignored.
        neglecting_max_coefficient: Coefficient threshold below which interactions are ignored.
        decompose_stop_number: Minimum remaining variables before switching to a classical solver.
        decompose_threshold: Cost-function threshold for accepting an embedded placement.
        decompose_break_placement: Minimum placed vertices required to form a valid subproblem.
        max_min_dist_ratio: Maximum allowed ratio between the largest and the smallest
            inter-atom distance in an extracted sub-problem's embedding.
    """

    neglecting_inter_distance: float = 1.5
    neglecting_max_coefficient: float = 1.0
    decompose_stop_number: int = 15
    decompose_threshold: float = 250
    decompose_break_placement: int = 3
    max_min_dist_ratio: float = float("inf")

    @staticmethod
    def _from_decomposition_config(
        config: DecompositionConfig, *, max_min_dist_ratio: float
    ) -> Config:
        """Create a :class:`Config` from a user-facing :class:`DecompositionConfig`."""
        return Config(
            neglecting_inter_distance=config.neglecting_inter_distance,
            neglecting_max_coefficient=config.neglecting_max_coefficient,
            decompose_stop_number=config.decompose_stop_number,
            decompose_threshold=config.decompose_threshold,
            decompose_break_placement=config.decompose_break_placement,
            max_min_dist_ratio=max_min_dist_ratio,
        )


class Instance(QUBOInstanceBase):
    """A QUBO instance augmented with decomposition state.

    Maintains the global solution, the working QUBO matrix, and the
    dictionaries of vertices still to place and already placed vertices.
    """

    def __init__(
        self,
        parent_instance: QUBOInstanceBase,
        *,
        config: Config = Config(),
    ):
        """Initialize the decomposition-aware QUBO instance.

        Args:
            parent_instance: The original QUBO instance.
            config: Decomposition algorithm parameters.
        """
        super().__init__(parent_instance.matrix)
        self._parent_instance = copy.deepcopy(parent_instance)

        self._global_solution: Bitstring = torch.full(
            (parent_instance.size,), -1, dtype=bitstring.dtype()
        )
        self._qubo_matrix: torch.Tensor = parent_instance.matrix.clone()
        self._vertices_to_place: dict[int, VertexToPlace] = {}
        self._placed_vertices: dict[int, WeightedZone] = {}
        self._decomposition: list[list[int]] = []

        dist_matrix = compute_distance_interaction_matrix(
            self._qubo_matrix,
            neglecting_inter_distance=config.neglecting_inter_distance,
            neglecting_max_coefficient=config.neglecting_max_coefficient,
        )
        # The following dictionary contain vertices to apply the decomposition search
        # where each vertex key maps to other blocking, separated and neighbors vertices
        # and gets updated as we iterate in the decomposition
        self._vertices_to_place = vertices_to_place(
            dist_matrix,
            self._qubo_matrix,
            separation_threshold=config.neglecting_inter_distance,
        )

        update(self, SubQUBOInstance(), Solution())


class SubQUBOInstance(QUBOInstanceBase):
    """A sub-problem extracted from a decomposed QUBO.

    Stores the sub-matrix together with a mapping from original variable
    indices to local (sub-problem) indices.
    """

    def __init__(
        self,
        coefficients: Matrix = matrix.zeros(0),
        map_index_vertices: dict[int, int] = {},
    ):
        """
        Args:
            coefficients: Square coefficient matrix for the sub-problem.
                Defaults to an empty matrix (used as a sentinel for "no
                sub-problem extracted").
            map_index_vertices: Mapping from original global variable indices
                to local (sub-problem) column/row indices.
        """
        super().__init__(coefficients)
        self._map_index_vertices = map_index_vertices


def extract_subqubo(
    qubo: Instance,
    config: Config,
    *,
    last: bool = False,
    rng: torch.Generator = torch_rng(),
) -> SubQUBOInstance:
    """Extract an embeddable sub-problem from the decomposed QUBO.

    Runs a geometric search to find a cluster of vertices that can be
    placed on the device, then builds the corresponding sub-matrix.

    Args:
        qubo: The decomposition-aware QUBO instance.
        config: Decomposition algorithm parameters.
        last: If ``True``, extract the remaining (final) sub-problem
            without geometric search.
        rng: Random number generator for reproducibility.

    Returns:
        A :class:`SubQUBOInstance` with the extracted sub-matrix and
        index mapping.  Returns an empty instance if the geometric
        search yields too few vertices.
    """
    if last:
        matrix_to_solve, map_index_vertices = last_target_matrix(
            list(qubo._vertices_to_place.keys()),
            qubo._qubo_matrix,
        )
        return SubQUBOInstance(matrix_to_solve, map_index_vertices)

    # find a first vertex to start the geometric search
    # random works better according to some performed numerics
    # sort to have reproducibility when setting the seed
    keys = sorted(qubo._vertices_to_place.keys())
    idx: int = int(torch.randint(0, len(keys), (), generator=rng).item())
    first_vertex_search = keys[idx]

    min_distance, max_radial_distance = compute_min_max_distances(
        qubo._qubo_matrix, max_min_dist_ratio=config.max_min_dist_ratio
    )
    qubo._placed_vertices = geometric_search(
        qubo._qubo_matrix,
        qubo._vertices_to_place,
        first_vertex_search,
        config.decompose_threshold,
        min_distance=min_distance,
        max_radial_distance=max_radial_distance,
        rng=rng,
    )
    if len(qubo._placed_vertices) <= config.decompose_break_placement:
        return SubQUBOInstance()

    matrix_to_solve, map_index_vertices = interaction_matrix_from_placed(qubo._placed_vertices)
    return SubQUBOInstance(matrix_to_solve, map_index_vertices)


def update(qubo: Instance, subqubo: SubQUBOInstance, subsolution: Solution) -> Solution:
    """Merge a sub-solution into the global state and return the full solution if complete.

    Records the solved sub-problem in the decomposition log, writes the
    best bitstring from *subsolution* into the corresponding positions of the
    global solution vector, propagates updated edge values to the remaining
    unplaced vertices, and greedily fixes any variable whose local field turns
    positive.

    Args:
        qubo: The decomposition-aware QUBO instance holding the mutable global
            state (global solution vector, working matrix, vertex dictionaries).
        subqubo: The sub-problem that was just solved.  An empty instance
            (no index mapping) is accepted and simply skips the merge step;
            this is used for the initial call that bootstraps the state.
        subsolution: The solution returned by the solver for *subqubo*.
            Only the best bitstring (index 0) is used.  An empty solution
            skips the write-back.

    Returns:
        A :class:`~qubosolver.types.Solution` containing the single
        complete global bitstring (with costs and probabilities computed) once
        all variables have been assigned, or an empty
        :class:`~qubosolver.types.Solution` if unassigned variables remain.
    """
    if subqubo._map_index_vertices:
        qubo._decomposition.append(list(subqubo._map_index_vertices.keys()))
    if subsolution:
        update_global_solution(
            global_solution=qubo._global_solution,
            sub_solution=subsolution.bitstrings[0],
            mapping=subqubo._map_index_vertices,
        )

    transfer_edge_values(
        qubo._vertices_to_place,
        qubo._placed_vertices,
        qubo._global_solution,
        qubo._qubo_matrix,
    )

    positive_vertices = positive_vertices_update(qubo._vertices_to_place, qubo._global_solution)

    for v in positive_vertices:
        qubo._decomposition.append([v])

    if (qubo._global_solution == -1).any():
        return Solution()

    # Probabilities and counts are ignored as we return one solution
    solution = Solution(
        bitstrings=qubo._global_solution.unsqueeze(0),
        counts=vectori.tensor([1]),
    )._update(qubo)

    return solution
