from __future__ import annotations

from functools import singledispatch

import numpy as np
import torch


@singledispatch
def convert_to_tensor(
    data: list | dict | tuple | np.ndarray | torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    raise TypeError(f"Unsupported data type: {type(data)}")


@convert_to_tensor.register
def from_dict(data: dict, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if all(isinstance(k, int) and isinstance(v, (int, float)) for k, v in data.items()):
        size = max(data.keys()) + 1
        tensor = torch.zeros(size, dtype=dtype, device=device)
        for index, value in data.items():
            tensor[index] = float(value)
        return tensor
    elif all(
        isinstance(k, tuple)
        and len(k) == 2
        and isinstance(k[0], int)
        and isinstance(k[1], int)
        and isinstance(v, (int, float))
        for k, v in data.items()
    ):
        max_index = max(max(i, j) for i, j in data.keys())
        size = max_index + 1
        tensor = torch.zeros((size, size), dtype=dtype, device=device)
        for (i, j), value in data.items():
            tensor[i, j] = float(value)
            if i != j:
                tensor[j, i] = float(value)
        return tensor
    raise TypeError(
        "Unsupported dictionary format: Expected dict[int, float] for 1D tensors "
        "or dict[Tuple[int, int], float] for 2D tensors."
    )


@convert_to_tensor.register
def from_list(data: list, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    array = np.asarray(data, dtype=float)
    return torch.tensor(array, dtype=dtype, device=device)


@convert_to_tensor.register
def from_tuple(
    data: tuple, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    array = np.asarray(data, dtype=float)
    return torch.tensor(array, dtype=dtype, device=device)


@convert_to_tensor.register
def from_ndarray(
    data: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)


@convert_to_tensor.register
def from_tensor(
    data: torch.Tensor, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return data.to(dtype=dtype, device=device)


def generate_symmetric_mask(
    size: int, target: int, device: str, generator: torch.Generator
) -> torch.Tensor:
    """Generate a symmetric boolean mask with an exact number of True elements
        to match a certain density of QUBO.
        Used in the `from_random` method of `QUBODataset`.

    Args:
        size (int): Size of problem.
        target (int): Target number of elements.
        device (str): Torch device.
        generator (torch.Generator): generator for randomness.

    Returns:
        torch.Tensor: Mask.
    """
    possible_x = []
    for x in range(1, min(size, target) + 1):
        if (target - x) % 2 == 0:
            y = (target - x) // 2
            if y <= (size * (size - 1)) // 2:
                possible_x.append(x)
    if not possible_x:
        x, y = 1, 0
    else:
        x = possible_x[
            torch.randint(0, len(possible_x), (1,), device=device, generator=generator).item()
        ]
        y = (target - x) // 2

    mask = torch.zeros((size, size), dtype=torch.bool, device=device)
    diag_indices = torch.randperm(size, device=device, generator=generator)[:x]
    for i in diag_indices.tolist():
        mask[i, i] = True

    upper_indices = torch.tensor(
        [(i, j) for i in range(size) for j in range(i + 1, size)],
        device=device,
    )
    if upper_indices.size(0) > 0 and y > 0:
        perm = torch.randperm(upper_indices.size(0), device=device, generator=generator)[:y]
        chosen_upper = upper_indices[perm]
        for i, j in chosen_upper.tolist():
            mask[i, j] = True
            mask[j, i] = True
    return mask
