from __future__ import annotations

import torch
from qubosolver.types import Matrix


def quadratic_cost(x: torch.Tensor, Q: Matrix) -> float:
    """Compute the quadratic cost for a single binary vector under a QUBO matrix.

    The cost is defined as the quadratic form :math:`x^T Q x`.

    Args:
        x: Binary tensor of shape ``(n,)``. Must be 1-dimensional.
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.

    Returns:
        float: The scalar cost value.

    Raises:
        ValueError: If ``x`` is not 1-dimensional. Use
            :func:`batched_quadratic_cost` for batched inputs.

    Example:
        >>> Q = torch.tensor([[1., -1.], [-1., 2.]])
        >>> x = torch.tensor([1., 0.])
        >>> quadratic_cost(x, Q)
        1.0
    """
    dtype = Q.dtype
    x_ = x.to(dtype)
    if x_.dim() != 1:
        raise ValueError("This method is for vector only. Use batched_quadratic_cost instead")
    return float(torch.linalg.multi_dot([x_, Q, x_]).item())


def batched_quadratic_cost(x: Matrix, Q: Matrix) -> Matrix:
    """Compute the quadratic cost for a batch of binary vectors under a QUBO matrix.

    The cost for each vector is defined as the quadratic form :math:`x_i^T Q x_i`.

    Args:
        x: Binary tensor of shape ``(b, n)``, where ``b`` is the batch size
            and ``n`` is the number of variables.
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.

    Returns:
        Matrix: A 1-D tensor of shape ``(b,)`` containing one cost value per
        input vector in the batch.

    Example:
        >>> Q = torch.tensor([[1., -1.], [-1., 2.]])
        >>> x = torch.tensor([[1., 0.], [0., 1.]])
        >>> batched_quadratic_cost(x, Q)
        tensor([1., 2.])
    """
    return torch.einsum("bi,ij,bj->b", x, Q, x)
