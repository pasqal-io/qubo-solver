from __future__ import annotations

import torch
from qubosolver.types import Matrix, Tensor


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
            `batched_quadratic_cost` for batched inputs.

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


def _flip_deltas(Q: Matrix, X: Tensor, QX: Tensor) -> Tensor:
    """Energy change from flipping each bit, for every run in the batch at once.

    For a single run, flipping $x_i \\to 1 - x_i$ changes $x^T Q x$ by
    $(1 - 2 x_i)(Q_{ii} + 2 (Qx)_i - 2 Q_{ii} x_i)$, which only needs row/column
    `i` of `Q` -- already summarized in ``QX = X @ Q``. Evaluating that for all
    `i` and all runs is a single elementwise expression over the ``(b, n)``
    tensors, replacing ``b * n`` scalar evaluations.

    Args:
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.
        X: Current binary states of shape ``(b, n)``, in `Q`'s dtype.
        QX: The product ``X @ Q``, of shape ``(b, n)``, maintained
            incrementally by the caller.

    Returns:
        A ``(b, n)`` tensor whose ``[r, i]`` entry is the energy change run `r`
            would see from flipping its bit `i`.
    """
    return 2.0 * QX * (1.0 - 2.0 * X) + Q.diagonal()
