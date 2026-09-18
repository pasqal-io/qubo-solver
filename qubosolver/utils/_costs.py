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


def _flip_deltas(
    Q: Matrix,
    X: Tensor,
    QX: Tensor,
    diagonal: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Energy change from flipping each bit, for every run in the batch at once.

    For a single run, flipping $x_i \\to 1 - x_i$ changes $x^T Q x$ by
    $(1 - 2 x_i)(Q_{ii} + 2 (Qx)_i - 2 Q_{ii} x_i)$, which only needs row/column
    `i` of `Q` -- already summarized in ``QX = X @ Q``. Evaluating that for all
    `i` and all runs is a single elementwise expression over the ``(b, n)``
    tensors, replacing ``b * n`` scalar evaluations.

    The expression is evaluated as ``2*QX - 4*QX*X + diag`` via a broadcast copy
    plus one fused multiply-add, which keeps the whole thing to a single
    ``(b, n)`` allocation (or none at all, when `out` is supplied) instead of the
    four temporaries the naive elementwise form materializes.

    Args:
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.
        X: Current binary states of shape ``(b, n)``, in `Q`'s dtype.
        QX: The product ``X @ Q``, of shape ``(b, n)``, maintained
            incrementally by the caller.
        diagonal: Optional precomputed ``Q.diagonal()`` of shape ``(n,)``.
            Callers in a loop should hoist it out and pass it in, so the
            diagonal view is not rebuilt on every call. Defaults to reading it
            from `Q`.
        out: Optional ``(b, n)`` buffer to write the result into, in `Q`'s
            dtype. Reused across iterations by callers in a loop to avoid
            reallocating the delta tensor every round. Must not alias `QX` or
            `X`. Defaults to allocating a fresh tensor.

    Returns:
        A ``(b, n)`` tensor whose ``[r, i]`` entry is the energy change run `r`
            would see from flipping its bit `i`. This is `out` itself when `out`
            is supplied.
    """
    diag = Q.diagonal() if diagonal is None else diagonal
    # dE = 2*QX*(1 - 2*X) + diag = (2*QX + diag) - 4*QX*X, built as one
    # broadcast add followed by one in-place fused multiply-add.
    dE = torch.add(QX, diag, alpha=0.5, out=out)
    dE *= 2.0
    return dE.addcmul_(QX, X, value=-4.0)
