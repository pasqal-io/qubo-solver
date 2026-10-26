"""Geometric partial embedding of QUBO instances.

Unlike the other embedders, which place every variable of an instance, the
geometric embedder places *what it can*: starting from a random variable, a
geometric search grows a cluster of variables whose QUBO couplings can be
reproduced by atom positions (interactions ``1 / r ** 6``) within the device
constraints. It returns the register together with the placed variables.

This makes it the building block of decomposition: embed part of the
instance, solve the sub-instance on those variables, fix them with
[`variable_fixing`][qubosolver.transforms.variable_fixing], and repeat on the
reduced instance:

```python
from qubosolver import embedding, solving
from qubosolver.transforms import variable_fixing

reduced = variable_fixing.apply_recursively(instance)  # the embedder needs diag < 0
while reduced.size > stop:
    register, variables = embedding.geometric.embed_for_device(reduced, device)
    if len(variables) <= min_atoms:
        break
    sub = reduced.subinstance(variables)
    sub_solution = solving.tabu_search.solve(sub)
    rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
    variable_fixing.apply(reduced, (rule,), inplace=True)  # candidate rule alone
    variable_fixing.apply_recursively(reduced, inplace=True)  # Hansen to a fixed point
solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)
```
"""

# TODO(abussy-pasqal): Revisit the module name (geometric vs partial / decomposition-oriented
# naming).

from __future__ import annotations

import logging
from dataclasses import dataclass

import qoolqit
import torch

from qubosolver import Instance, tensor, torch_rng
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal
from qubosolver.utils.quantum import _max_min_distance_ratio

from ._algorithms.geometric import (
    compute_distance_interaction_matrix,
    compute_min_max_distances,
    geometric_search,
    vertices_to_place,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the geometric partial embedding algorithm.

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
    rng: torch.Generator | None = None,
) -> tuple[qoolqit.Register, list[int]]:
    """Embed part of a QUBO instance with the geometric search, sized for *device*.

    Convenience wrapper around [`embed`][] that derives `Config.max_min_dist_ratio`
    from *device* via [`Config.from_device`][].

    Args:
        instance: The QUBO instance to partially embed.
        device: Target quantum device the resulting register must fit.
        rng: Random number generator choosing the first variable and driving the
            search. Defaults to `torch_rng("global")`.

    Returns:
        The register and the placed variables; see [`embed`][].
    """
    logger.debug("embed_for_device: instance size=%d, device=%r", instance.size, device)
    return embed(instance, config=Config.from_device(device), rng=rng)


def embed(
    instance: Instance,
    *,
    config: Config | None = None,
    rng: torch.Generator | None = None,
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
            coefficients must be non-negative and diagonal coefficients
            negative.
        config: Geometric search parameters (see [`Config.from_device`][] for
            deriving `max_min_dist_ratio` from a device).
        rng: Random number generator choosing the first variable and driving the
            search. Defaults to `torch_rng("global")`, so that it follows
            `torch.manual_seed`.

    Returns:
        A tuple `(register, variables)`: qubit *i* of `register` (labelled *i*)
            embeds variable `variables[i]` of `instance`. `variables` is sorted,
            so `instance.subinstance(variables)` is the sub-instance in
            register order.

    Raises:
        ValueError: If `instance` has no variables, has negative off-diagonal
            coefficients, or has non-negative diagonal coefficients (fix them
            first with
            [`variable_fixing.apply_recursively`][qubosolver.transforms.variable_fixing.apply_recursively]).
            With non-negative off-diagonal coefficients, a variable with a
            non-negative diagonal coefficient is optimally 0, and Hansen's rule
            fixes it to 0.
        ValueError: If `instance` is diagonal with at least 2 variables: there is
            no coupling to reproduce, and its optimum should be read off the diagonal with
            [`trivial_solution_search.solve`][qubosolver.solving.classical.trivial_solution_search.solve].
    """
    config = config or Config()
    rng = rng or torch_rng("global")
    logger.debug("embed: instance size=%d, config=%r", instance.size, config)
    if not instance:
        msg = "Cannot embed an empty instance (size=0): nothing to place."
        raise ValueError(msg)

    if _has_negative_offdiagonal(instance.matrix):
        msg = "QUBOs with negative off-diagonal coefficients cannot be embedded."
        raise ValueError(msg)

    if non_negative := (instance.matrix.diagonal() >= 0).nonzero().flatten().tolist():
        msg = (
            f"Variables {non_negative} have non-negative diagonal coefficients, but the geometric "
            "embedder requires negative ones. Fix these variables first, e.g. with "
            "`variable_fixing.apply_recursively` (Hansen's rule fixes them to 0)."
        )
        raise ValueError(msg)

    if instance.size == 1:
        # A single atom has no off-diagonal term to place it relative to,
        # so it is placed at the origin without running the search.
        return qoolqit.Register.from_coordinates(tensor.zeros(1, 2)), [0]

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
    first_vertex = int(torch.randint(0, instance.size, (), generator=rng).item())

    # TODO(abussy-pasqal): The distance scale comes from the largest off-diagonal coefficient
    # of `instance`. In a decomposition loop, that is the reduced instance, so the scale changes
    # between rounds as the strongest couplings get fixed (the former stateful decomposition
    # kept the scale of the original instance for the whole run). Decide which scale to use;
    # part of the known scale dependence of the decomposition.
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
    coords = tensor.tensor([[placed[v].x, placed[v].y] for v in variables])
    return qoolqit.Register.from_coordinates(coords), variables
