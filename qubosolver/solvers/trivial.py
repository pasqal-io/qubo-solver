"""Trivial QUBO solution detection.

Provides `trivial_solution_search`, which recognises coefficient
patterns whose optimal solution can be read off analytically without any
search.
"""
from __future__ import annotations

import torch

from qubosolver import Instance, Solution, bitstring, vector, vectori


def trivial_solution_search(instance: Instance) -> Solution:
    """Analytically solve a QUBO when the coefficient structure is trivial.

    Three patterns are recognised:

    1. **All coefficients ≥ 0** — the all-zeros bitstring ``0^n`` is optimal.
    2. **All coefficients ≤ 0** — the all-ones bitstring ``1^n`` is optimal.
    3. **Diagonal matrix** — each variable is independent; bits with a
       negative diagonal entry are set to 1, the rest to 0.

    Args:
        instance (Instance): The QUBO problem whose matrix is inspected.

    Returns:
        Solution: A single-bitstring solution when a trivial case is
            detected, or an empty :class:`~qubosolver.types.Solution`
            (no bitstrings) when none of the three patterns apply.
    """
    coeffs = instance.matrix
    n = instance.size

    # Case 1: all coeffs >= 0 → x = [0,...,0]
    if torch.all(coeffs >= 0):
        raw = bitstring.zeros(n)
        # always make a batch of one: shape (1, n)
        batch = raw.unsqueeze(0)
        cost = instance.evaluate_solution(raw)
        return Solution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    # Case 2: all coeffs <= 0 → x = [1,...,1]
    if torch.all(coeffs <= 0):
        raw = torch.ones(n, dtype=bitstring.dtype())
        # always make a batch of one: shape (1, n)
        batch = raw.unsqueeze(0)
        cost = instance.evaluate_solution(raw)
        return Solution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    # Case 3: diagonal cases
    # negative coeffs gets 1, positive gets 0
    diagonal = torch.diag(coeffs)
    if (torch.diag(diagonal) == coeffs).all():
        raw = (diagonal < 0).to(bitstring.dtype())
        cost = instance.evaluate_solution(raw)
        batch = raw.unsqueeze(0)
        return Solution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    return Solution()
