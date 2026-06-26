from __future__ import annotations

import torch

from qubosolver import QUBOInstance, QUBOSolution, bitstring, vector, vectori


def trivial_solution_search(Q: QUBOInstance) -> QUBOSolution:
    """
    Check for the two trivial QUBO cases:
        1) all coefficients >= 0  → solution = 0^n
        2) all coefficients <= 0  → solution = 1^n
        3) diagonal qubo,  negative coeffs gets 1, positive gets 0

    Returns:
        QUBOSolution if a trivial case applies, else None.
    """
    coeffs = Q.matrix
    n = Q.size

    # Case 1: all coeffs >= 0 → x = [0,...,0]
    if torch.all(coeffs >= 0):
        raw = bitstring.zeros(n)
        # always make a batch of one: shape (1, n)
        batch = raw.unsqueeze(0)
        cost = Q.evaluate_solution(raw)
        return QUBOSolution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    # Case 2: all coeffs <= 0 → x = [1,...,1]
    if torch.all(coeffs <= 0):
        raw = torch.ones(n, dtype=bitstring.dtype())
        # always make a batch of one: shape (1, n)
        batch = raw.unsqueeze(0)
        cost = Q.evaluate_solution(raw)
        return QUBOSolution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    # Case 3: diagonal cases
    # negative coeffs gets 1, positive gets 0
    diagonal = torch.diag(coeffs)
    if (torch.diag(diagonal) == coeffs).all():
        raw = (diagonal < 0).to(bitstring.dtype())
        cost = Q.evaluate_solution(raw)
        batch = raw.unsqueeze(0)
        return QUBOSolution(
            bitstrings=batch,
            counts=vectori.tensor([1]),
            costs=vector.tensor([cost]),
        )

    return QUBOSolution()
