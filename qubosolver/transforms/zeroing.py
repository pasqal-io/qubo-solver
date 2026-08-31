"""Zeroing fallback for QUBO negative off-diagonal coefficients.

Bit-flip preprocessing (see
[`transforms.negative_bitflip`][qubosolver.transforms.negative_bitflip]) removes as much negative
off-diagonal weight as possible, but some negative coefficients may remain when
the problem is not fully bipartisable. Quantum (Rydberg) solvers cannot embed
such coefficients, so this module offers a last-resort approximation:
[`apply`][] sets every remaining negative
off-diagonal coefficient to zero and records which positions were zeroed in a
zeroing [`Instance`][qubosolver.transforms.zeroing.Instance].

```python
from qubosolver.transforms import negative_bitflip, zeroing

reduced_instance = negative_bitflip.apply(instance, time_limit_s=60.0)
zeroed_instance = zeroing.apply(reduced_instance)  # drop any negative coefficient bit flips could not remove
print(zeroed_instance.zeroed_edges)       # (N, 2) tensor of zeroed (i, j) index pairs
```
"""

from __future__ import annotations

import copy

import torch

import qubosolver
from qubosolver.types import Matrix, Solution, Vectori, vector, vectori


class Instance(qubosolver.Instance):
    """A QUBO [`Instance`][qubosolver.Instance] recording zeroing history.

    Records *which* off-diagonal coefficients were set to zero by
    [`apply`][qubosolver.transforms.zeroing.apply] by keeping the matrix of the
    removed negative coefficients ([`negative_matrix`][]) rather than a single
    flag.  Because the QUBO matrix is symmetric, each zeroed interaction appears
    twice in [`negative_matrix`][] but once in
    [`zeroed_edges`][].
    """

    def __init__(self, parent_instance: qubosolver.Instance):
        """Initialize from a QUBO instance, before any zeroing.

        Args:
            parent_instance: The QUBO instance to extend with zeroing state.
        """
        super().__init__(parent_instance.matrix.detach().clone())
        self._parent_instance = copy.deepcopy(parent_instance)
        self.negative_matrix: Matrix = torch.zeros_like(self._matrix)
        """Matrix of removed negative coefficients: same (symmetric) shape as the
        QUBO matrix, holding the original values at zeroed positions and 0 elsewhere."""

    @property
    def zeroed_edges(self) -> Vectori:
        """The zeroed interactions as an ``(N, 2)`` tensor of ``(i, j)`` index pairs.

        Each symmetric pair is reported once (``i < j``); ``N`` is the number of
        zeroed off-diagonal interactions.
        """
        upper = torch.triu(self.negative_matrix != 0, diagonal=1)
        return vectori.as_tensor(upper.nonzero())


def apply(instance: qubosolver.Instance) -> Instance:
    """Set remaining negative off-diagonal coefficients to zero.

    Approximates the QUBO by dropping any negative off-diagonal coefficient that
    bit flips could not remove, so a quantum solver can embed it.  Returns a
    [`Instance`][qubosolver.transforms.zeroing.Instance] whose [`negative_matrix`][qubosolver.transforms.zeroing.Instance.negative_matrix]
    holds the removed coefficients (an all-zero matrix when nothing was zeroed).

    Args:
        instance: The QUBO instance to zero.

    Returns:
        A zeroing instance.
    """
    zeroed_instance = Instance(instance)

    Q = instance.matrix
    n = Q.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=Q.device)
    negative_mask = offdiag_mask & (Q < 0)

    zeroed_instance.negative_matrix[negative_mask] = Q[negative_mask]
    zeroed_instance._matrix[negative_mask] = 0.0

    return zeroed_instance


def lift(zeroed_solution: Solution, zeroed_instance: Instance) -> Solution:
    """Map a solution of the zeroed QUBO back onto the pre-zeroing problem.

    Zeroing only drops coefficients; it does not rename or remove variables, so
    the bitstrings are carried over unchanged.  Costs are recomputed against the
    pre-zeroing matrix (``_parent_instance``) so they reflect the true, non-
    approximated objective rather than the zeroed one.  When nothing was zeroed,
    returns a deep copy of `zeroed_solution` unchanged.

    Args:
        zeroed_solution: Solution obtained on the zeroed QUBO.
        zeroed_instance: The zeroed instance produced by [`apply`][].

    Returns:
        A new solution with costs evaluated against the pre-zeroing matrix.
    """
    if not zeroed_instance.zeroed_edges.numel():
        return copy.deepcopy(zeroed_solution)

    solution = Solution()
    solution.bitstrings = zeroed_solution.bitstrings
    solution.costs = vector.tensor(
        [zeroed_instance._parent_instance.cost(b) for b in solution.bitstrings]
    )
    solution.counts = zeroed_solution.counts
    solution.probabilities = zeroed_solution.probabilities

    return solution
