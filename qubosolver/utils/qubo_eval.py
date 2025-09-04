from __future__ import annotations

import torch


def calculate_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float:
    """Apply the default qubo evaluation b Q b^T.

    Args:
        bitstring (str): Candidate bitstring.
        QUBO (torch.Tensor): QUBO coefficients.

    Returns:
        float: Evaluation.
    """

    lb = [int(b) for b in list(bitstring)]
    z = torch.tensor(lb, dtype=QUBO.dtype)
    qz = torch.matmul(QUBO, z)
    res = torch.dot(z, qz).item()
    return float(res)
