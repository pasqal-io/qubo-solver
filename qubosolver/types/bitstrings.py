from __future__ import annotations

from typing import Any
import torch
from . import linalg, bitstring


def dtype() -> torch.dtype:
    """Returns the dtype used for bitstrings (``torch.int8``)."""
    return bitstring.dtype()


def device() -> torch.device:
    """Returns the globally configured torch device."""
    return bitstring.device()


def zeros(count: int, n_bits: int, *, device: torch.device = device()) -> linalg.Bitstrings:
    """Creates a zero-filled 2-D bitstrings tensor.

    Args:
        count: Number of bitstrings (rows).
        n_bits: Length of each bitstring (columns).
        device: Torch device for the tensor.

    Returns:
        A 2-D ``int8`` tensor of shape ``(count, n_bits)``.
    """
    return torch.zeros((count, n_bits), dtype=dtype(), device=device)


def tensor(data: Any, *, device: torch.device = device(), **kwargs: Any) -> linalg.Bitstrings:
    """Creates a 2-D bitstrings tensor from the given data.

    Args:
        data: Input data (nested list or array-like of 0s and 1s).
        device: Torch device for the tensor.
        **kwargs: Extra keyword arguments forwarded to :func:`torch.tensor`.

    Returns:
        A 2-D ``int8`` tensor.
    """
    return torch.tensor(data, dtype=dtype(), device=device, **kwargs)


def from_torch(tensor: torch.Tensor) -> linalg.Bitstrings:
    """Converts an existing torch tensor to bitstrings (``int8``, on the global device).

    Args:
        tensor: Source tensor to convert.

    Returns:
        The tensor cast to ``int8`` on the global device.
    """
    return tensor.to(dtype=dtype(), device=device())


def from_strings(strings: list[str], *, device: torch.device = device()) -> linalg.Bitstrings:
    """Creates a 2D Bitstrings tensor from a list of bitstring strings."""
    if len(strings) == 0:
        return zeros(0, 0, device=device)
    lengths = {len(s) for s in strings}
    if len(lengths) != 1:
        raise ValueError(
            f"All bitstrings must have the same length, got lengths: {sorted(lengths)}"
        )
    return torch.stack([bitstring.from_string(s, device=device) for s in strings])


def to_strings(bitstrings: linalg.Bitstrings) -> list[str]:
    """Converts a 2D Bitstrings tensor into a list of bitstring strings."""
    return [bitstring.to_string(b) for b in bitstrings]
