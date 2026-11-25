# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Partial greedy embedding of QUBO instances.

Unlike the other embedders, which place every variable of an instance, the
partial greedy embedder places *what it can*: starting from a random variable, a
geometric search grows a cluster of variables whose QUBO couplings can be
reproduced by atom positions (interactions ``1 / r ** 6``) within the device
constraints. It returns the register together with the placed variables.

Warning:
    The partial greedy embedder and its search algorithms are in the prototype
    stage and may or may not change in the future.

This makes it the building block of decomposition: embed part of the
instance, solve the QUBO the register implements on those variables (its
couplings from the atom positions, its diagonal from the sub-instance, see
[`extract_qubo`][qubosolver.extract_qubo]), fix them with
[`variable_fixing`][qubosolver.transforms.variable_fixing], and repeat on the
reduced instance:

```python
from qubosolver import embedding, extract_qubo, solving
from qubosolver.transforms import variable_fixing

reduced = variable_fixing.apply_recursively(instance)  # the embedder needs diag < 0
while reduced.size > stop:
    register, variables = embedding.partial_greedy.embed_for_device(reduced, device)
    sub = extract_qubo(register, reduced.subinstance(variables))
    sub_solution = solving.tabu_search.solve(sub)
    variable_fixing.apply_candidate(reduced, sub_solution[0], variables, inplace=True)
    variable_fixing.apply_recursively(reduced, inplace=True)  # Hansen to a fixed point
solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import qoolqit

from qubosolver import Instance, numpy_rng
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal
from qubosolver.utils.quantum import _max_min_distance_ratio

from ._algorithms.partial_greedy import (
    compute_distance_interaction_matrix,
    compute_min_max_distances,
    geometric_search,
    vertices_to_place,
)

logger = logging.getLogger(__name__)

_PROTOTYPE_NOTICE = (
    "The partial greedy embedder and its search algorithms are in the prototype stage and may or "
    "may not change in the future."
)


@dataclass
class Config:
    """Configuration for the partial greedy embedding algorithm.

    Use `Config.from_device` to derive `max_min_dist_ratio` from a device's
    constraints instead of setting it by hand.

    Warning:
        `max_min_dist_ratio` is an advanced parameter: set it manually (or
        overwrite it after calling `from_device`) with care, since a bad value
        can produce a register that the target device cannot realize.

    Attributes:
        threshold: Cost-function threshold for accepting the placement of a
            variable, comparing its target QUBO couplings with the
            interactions of its position.
        neglecting_inter_distance: Distance assigned to neglected interactions
            in the distance interaction matrix.
        neglecting_max_coefficient: QUBO coefficient below which an
            interaction is neglected.
        max_min_dist_ratio: Maximum allowed ratio between the maximum radial
            distance and the minimum inter-atom distance in the resulting
            register.
    """

    threshold: float = 250.0
    neglecting_inter_distance: float = 1.5
    neglecting_max_coefficient: float = 1.0
    max_min_dist_ratio: float = float("inf")

    @staticmethod
    def from_device(device: qoolqit.Device) -> Config:
        """Create a [`Config`][] with `max_min_dist_ratio` derived from *device*.

        All fields can be overwritten on the returned instance.

        Args:
            device: Target quantum device to derive the distance constraints from.

        Returns:
            A configuration with a device-derived `max_min_dist_ratio`.
        """
        return Config(max_min_dist_ratio=_max_min_distance_ratio(device))


def embed_for_device(
    instance: Instance,
    device: qoolqit.Device,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[qoolqit.Register, list[int]]:
    """Embed part of a QUBO instance with the geometric search, sized for *device*.

    Convenience wrapper around [`embed`][] that derives `Config.max_min_dist_ratio`
    from *device* via [`Config.from_device`][].

    Args:
        instance: The QUBO instance to partially embed.
        device: Target quantum device the resulting register must fit.
        rng: Random number generator choosing the first variable and driving the
            search. Defaults to `numpy_rng("global")`.

    Returns:
        The register and the placed variables; see [`embed`][].
    """
    logger.debug("embed_for_device: instance size=%d, device=%r", instance.size, device)
    return embed(instance, config=Config.from_device(device), rng=rng)


def embed(
    instance: Instance,
    *,
    config: Config | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[qoolqit.Register, list[int]]:
    """Embed part of a QUBO instance with the geometric search.

    Starts from a randomly chosen variable, placed at the origin, and grows a
    cluster of variables whose couplings can be reproduced by atom positions
    (interactions ``1 / r ** 6``, adimensional units). At least one variable is
    always placed.

    Warning:
        A poorly chosen `config` can produce a register that is incompatible
        with a target device. See [`Config`][].

    Args:
        instance: The QUBO instance to partially embed. Off-diagonal
            coefficients must be non-negative and, with at least 2 variables,
            diagonal coefficients non-positive.
        config: Geometric search parameters (see [`Config.from_device`][] for
            deriving `max_min_dist_ratio` from a device).
        rng: Random number generator choosing the first variable and driving the
            search. Defaults to `numpy_rng("global")`, so that it follows
            `numpy.random.seed`.

    Returns:
        A tuple `(register, variables)`: qubit *i* of `register` (labelled *i*)
            embeds variable `variables[i]` of `instance`. `variables` is sorted,
            so `instance.subinstance(variables)` is the sub-instance in
            register order.

    Raises:
        ValueError: If `instance` has no variables, has negative off-diagonal
            coefficients, or has at least 2 variables and positive diagonal
            coefficients (fix them first with
            [`variable_fixing.apply_recursively`][qubosolver.transforms.variable_fixing.apply_recursively]).
            With non-negative off-diagonal coefficients, a variable with a
            positive diagonal coefficient is optimally 0, and Hansen's rule
            fixes it to 0.
        ValueError: If `instance` is diagonal with at least 2 variables: there is
            no coupling to reproduce, and its optimum should be read off the diagonal with
            [`trivial_solution_search.solve`][qubosolver.solving.classical.trivial_solution_search.solve].
    """
    logger.info(_PROTOTYPE_NOTICE)
    config = config or Config()
    rng = rng or numpy_rng("global")
    logger.debug("embed: instance size=%d, config=%r", instance.size, config)
    if not instance:
        msg = "Cannot embed an empty instance (size=0): nothing to place."
        raise ValueError(msg)

    if _has_negative_offdiagonal(instance.matrix):
        msg = "QUBOs with negative off-diagonal coefficients cannot be embedded."
        raise ValueError(msg)

    if instance.size == 1:
        # A single atom has no off-diagonal term to place it relative to,
        # so it is placed at the origin without running the search.
        return qoolqit.Register.from_coordinates(np.zeros((1, 2))), [0]

    if positive := np.flatnonzero(instance.matrix.diagonal() > 0).tolist():
        msg = (
            f"Variables {positive} have positive diagonal coefficients, but the partial greedy "
            "embedder requires non-positive ones. Fix these variables first, e.g. with "
            "`variable_fixing.apply_recursively` (Hansen's rule fixes them to 0)."
        )
        raise ValueError(msg)

    if instance.is_diagonal():
        msg = (
            "Cannot embed a diagonal instance: there is no coupling to reproduce. Its optimum is "
            "read off the diagonal: use `solving.trivial_solution_search.solve`."
        )
        raise ValueError(msg)

    qubo_matrix = instance.matrix
    dist_matrix = compute_distance_interaction_matrix(
        qubo_matrix,
        neglecting_inter_distance=config.neglecting_inter_distance,
        neglecting_max_coefficient=config.neglecting_max_coefficient,
    )
    vertices = vertices_to_place(
        dist_matrix, qubo_matrix, separation_threshold=config.neglecting_inter_distance
    )

    # A random first variable works better according to some performed numerics.
    first_vertex = int(rng.integers(0, instance.size))

    # The distance scale comes from the largest off-diagonal coefficient of `instance`. In a
    # decomposition loop, that is the reduced instance, so each round gets its own scale. This is
    # deliberate: a per-round scale leaves more room for a good embedding than one shared scale.
    min_distance, max_radial_distance = compute_min_max_distances(
        qubo_matrix, max_min_dist_ratio=config.max_min_dist_ratio
    )
    placed = geometric_search(
        qubo_matrix,
        vertices,
        first_vertex,
        config.threshold,
        min_distance=min_distance,
        max_radial_distance=max_radial_distance,
        rng=rng,
    )

    variables = sorted(placed)
    coords = np.asarray([[placed[v].x, placed[v].y] for v in variables], dtype=np.float64)
    return qoolqit.Register.from_coordinates(coords), variables
