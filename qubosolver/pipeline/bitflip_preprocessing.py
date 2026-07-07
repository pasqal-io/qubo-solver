from __future__ import annotations

import torch


def has_negative_offdiagonal(Q: torch.Tensor, eps: float = 0.0) -> bool:
    """Return True if Q has at least one negative off-diagonal coefficient."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=Q.device)
    return bool(torch.any(Q[offdiag_mask] < -eps).item())


def apply_bitflips_to_bitstrings(
    bitstrings: torch.Tensor,
    flips: torch.Tensor,
) -> torch.Tensor:
    """Apply or undo bit flips on bitstrings.

    The operation is its own inverse.
    """
    if bitstrings.numel() == 0:
        return bitstrings

    flips = flips.to(device=bitstrings.device, dtype=bitstrings.dtype)

    if bitstrings.ndim == 1:
        return torch.abs(bitstrings - flips)

    if bitstrings.ndim == 2:
        return torch.abs(bitstrings - flips.unsqueeze(0))

    raise ValueError("bitstrings must be a 1D or 2D tensor.")


def transform_qubo_by_bitflips(
    Q: torch.Tensor,
    flips: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Transform QUBO coefficients after variable bit flips.

    Convention:
        x_i = y_i       if flips_i = 0
        x_i = 1 - y_i   if flips_i = 1

    The returned offset satisfies:
        x^T Q x = y^T Q_flipped y + offset
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q.shape[0]
    if flips.numel() != n:
        raise ValueError("flips must have the same length as Q size.")

    dtype = Q.dtype
    device = Q.device

    f = flips.to(device=device, dtype=dtype).reshape(n)
    s = 1.0 - 2.0 * f

    Q_flipped = Q * torch.outer(s, s)

    linear = 2.0 * s * (Q @ f)
    Q_flipped = Q_flipped.clone()
    diag_idx = torch.arange(n, device=device)
    Q_flipped[diag_idx, diag_idx] += linear

    offset = float(f @ Q @ f)

    return Q_flipped, offset