from __future__ import annotations

import torch


def qubo_cost(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute the quadratic cost of a given binary vector under a QUBO matrix.

    The cost is defined as the quadratic form :math:`x^T Q x`.

    Args:
        x: Binary tensor of shape (n,) or (n, 1).
        Q: Symmetric QUBO coefficient matrix of shape (n, n).

    Returns:
        A scalar tensor representing the cost value.

    Example:
        >>> Q = torch.tensor([[1., -1.], [-1., 2.]])
        >>> x = torch.tensor([1., 0.])
        >>> qubo_cost(x, Q)
        tensor(1.)
    """
    x = x.to(torch.float32)
    Q = Q.to(torch.float32)
    return x.T @ Q @ x


def calculate_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float:
    """Apply the default qubo evaluation b Q b^T.

    Args:
        bitstring (str): Candidate bitstring.
        QUBO (torch.Tensor): QUBO coefficients.

    Returns:
        float: Evaluation.
    """

    lb = torch.tensor([int(b) for b in list(bitstring)])
    res = qubo_cost(lb, QUBO)
    return float(res)
