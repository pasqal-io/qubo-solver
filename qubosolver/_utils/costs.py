from __future__ import annotations

import torch
from qubosolver.types import Matrix


def quadratic_cost(x: torch.Tensor, Q: Matrix) -> float:
    dtype = Q.dtype
    x_ = x.to(dtype)
    if x_.dim() != 1:
        raise ValueError("This method is for vector only. Use batched_quadratic_cost instead")
    return float(torch.linalg.multi_dot([x_, Q, x_]).item())


def batched_quadratic_cost(x: Matrix, Q: Matrix) -> Matrix:
    """
    Compute the quadratic cost of a given binary vector under a QUBO matrix.

    The cost is defined as the quadratic form :math:`x^T Q x`.

    Args:
        x: Binary tensor of shape (n,) or (n, 1), or (b,n) if batched.
        Q: Symmetric QUBO coefficient matrix of shape (n, n).

    Returns:
        A scalar tensor representing the cost value.

    Example:
        >>> Q = torch.tensor([[1., -1.], [-1., 2.]])
        >>> x = torch.tensor([1., 0.])
        >>> qubo_cost(x, Q)
        tensor(1.)
    """
    return torch.einsum("bi,ij,bj->b", x, Q, x)
