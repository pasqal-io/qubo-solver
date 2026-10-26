"""Trivial QUBO solution detection.

Recognizes coefficient patterns whose optimal solution can be read off analytically without any
search.
"""

from __future__ import annotations

from qubosolver import Instance, Solution
from qubosolver.types.instance import _trivial_optimum


def solve(instance: Instance) -> Solution:
    """Solve a QUBO when the coefficient structure is trivial.

    Three patterns are recognised (see [`Instance.is_trivial`][qubosolver.Instance.is_trivial]):

    1. **All coefficients ≥ 0** — the all-zeros bitstring ``0^n`` is optimal.
    2. **All coefficients ≤ 0** — the all-ones bitstring ``1^n`` is optimal.
    3. **Diagonal matrix** — each variable is independent; bits with a
       negative diagonal entry are set to `1`, the rest to `0`.

    Args:
        instance: The QUBO problem whose matrix is inspected.

    Returns:
        A single-bitstring solution when a trivial case is
            detected, or an empty solution (no bitstrings) when none of the three patterns apply.
    """
    optimum = _trivial_optimum(instance)
    if optimum is None:
        return Solution()
    return Solution(bitstrings=optimum.unsqueeze(0))._update(instance, unit_counts=True)
