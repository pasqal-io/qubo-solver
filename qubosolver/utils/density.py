from __future__ import annotations

import torch

from qubosolver.qubo_types import DensityType

# Density thresholds
SPARSE_THRESHOLD: tuple[float, float] = (0.0, 0.3)
MEDIUM_THRESHOLD: tuple[float, float] = (0.3, 0.7)
HIGH_THRESHOLD: tuple[float, float] = (0.7, 1.0)


def classify_density(density: float) -> DensityType:
    """
    Classifies the density of a QUBO problem based on predefined thresholds.

    Args:
        density (float):
            The density value to classify. Should be in the range [0.0, 1.0].

    Returns:
        DensityType:
            The classification of the density (SPARSE, MEDIUM, or HIGH).
    """
    if SPARSE_THRESHOLD[0] <= density < SPARSE_THRESHOLD[1]:
        return DensityType.SPARSE
    elif MEDIUM_THRESHOLD[0] <= density < MEDIUM_THRESHOLD[1]:
        return DensityType.MEDIUM
    elif HIGH_THRESHOLD[0] <= density <= HIGH_THRESHOLD[1]:
        return DensityType.HIGH
    else:
        raise ValueError(f"Density {density} is outside the defined thresholds.")


def calculate_density(coefficients: torch.Tensor | None, size: int | None) -> float:
    """
    Calculates the density of a QUBO coefficient matrix.

    Density is defined as the fraction of non-zero elements in the matrix.

    Args:
        coefficients (torch.Tensor | None):
            Tensor representing the QUBO coefficients. If None, density is 0.
        size (int | None):
            Size of the QUBO matrix (number of rows/columns).

    Returns:
        float:
            The density value, ranging from 0.0 (completely sparse) to 1.0 (completely dense).
    """
    if coefficients is None or coefficients.numel() == 0:
        return 0.0

    total_elements = size**2 if size else 0
    non_zero_elements = torch.count_nonzero(coefficients).item()
    return float(non_zero_elements / total_elements)
