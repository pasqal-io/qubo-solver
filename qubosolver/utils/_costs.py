# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def quadratic_cost(x: NDArray[np.int8], Q: NDArray[np.float64]) -> float:
    """Compute the quadratic cost for a single binary vector under a QUBO matrix.

    The cost is defined as the quadratic form :math:`x^T Q x`.

    Args:
        x: Binary array of shape ``(n,)``. Must be 1-dimensional.
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.

    Returns:
        float: The scalar cost value.

    Raises:
        ValueError: If ``x`` is not 1-dimensional. Use
            `batched_quadratic_cost` for batched inputs.

    Example:
        >>> Q = np.array([[1., -1.], [-1., 2.]])
        >>> x = np.array([1., 0.])
        >>> quadratic_cost(x, Q)
        1.0
    """
    dtype = Q.dtype
    x_ = x.astype(dtype)
    if x_.ndim != 1:
        msg = "This method is for vector only. Use batched_quadratic_cost instead"
        raise ValueError(msg)
    return float(x_ @ Q @ x_)


def batched_quadratic_cost(x: NDArray[np.float64], Q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the quadratic cost for a batch of binary vectors under a QUBO matrix.

    The cost for each vector is defined as the quadratic form :math:`x_i^T Q x_i`.

    Args:
        x: Binary array of shape ``(b, n)``, where ``b`` is the batch size
            and ``n`` is the number of variables.
        Q: Symmetric QUBO coefficient matrix of shape ``(n, n)``.

    Returns:
        A 1-D array of shape ``(b,)`` containing one cost value per
            input vector in the batch.

    Example:
        >>> Q = np.array([[1., -1.], [-1., 2.]])
        >>> x = np.array([[1., 0.], [0., 1.]])
        >>> batched_quadratic_cost(x, Q)
        array([1., 2.])
    """
    x_ = x.astype(Q.dtype, copy=False)
    # Row-wise dot of (x @ Q) with x, without materializing the (b, n) product:
    # `np.sum((x @ Q) * x, axis=1)` allocates one more large temporary per call, which
    # for big float64 batches is returned to the OS and page-faulted back in every time.
    costs: NDArray[np.float64] = np.einsum("bi,bi->b", x_ @ Q, x_)
    return costs


def _flip_deltas(
    Q: NDArray[np.float64],
    X: NDArray[np.float64],
    QX: NDArray[np.float64],
    diagonal: NDArray[np.float64] | None = None,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    r"""Energy change from flipping each bit, for every run in the batch at once.

    For a single run, flipping $x_i \\to 1 - x_i$ changes $x^T Q x$ by
    $(1 - 2 x_i)(Q_{ii} + 2 (Qx)_i - 2 Q_{ii} x_i)$, which only needs row/column
    `i` of `Q` -- already summarized in ``QX = X @ Q``. Evaluating that for all
    `i` and all runs is a single elementwise expression over the ``(b, n)``
    arrays, replacing ``b * n`` scalar evaluations.

    The expression is evaluated as ``2*QX + diag - 4*QX*X``, writing into `out`
    when it is supplied.

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
            reallocating the delta array every round. Must not alias `QX` or
            `X`. Defaults to allocating a fresh array.

    Returns:
        A ``(b, n)`` array whose ``[r, i]`` entry is the energy change run `r`
            would see from flipping its bit `i`. This is `out` itself when `out`
            is supplied.
    """
    diag = Q.diagonal() if diagonal is None else diagonal
    # dE = 2*QX*(1 - 2*X) + diag = (2*QX + diag) - 4*QX*X
    dE: NDArray[np.float64] = np.multiply(QX, 2.0, out=out)
    dE += diag
    dE -= 4.0 * QX * X
    return dE
