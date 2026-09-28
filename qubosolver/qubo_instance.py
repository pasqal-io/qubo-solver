from __future__ import annotations

import torch
from numpy.typing import ArrayLike

from .data import QUBOSolution
from .data_utils import convert_to_tensor
from .qubo_types import DensityType
from .utils.density import (
    calculate_density,
    classify_density,
)


class QUBOInstance:
    """
    Represents a single instance of a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Attributes:
        coefficients (torch.Tensor):
            Tensor of shape (size, size), representing the QUBO coefficients.
        device (str):
            Device where tensors are allocated (e.g., "cpu" or "cuda").
        dtype (torch.dtype):
            Data type of the tensors (e.g., torch.float32).
        size (int | None):
            Size of the QUBO matrix (number of variables).
        solution (QUBOSolution | None):
            Solution to the QUBO problem, if available.
        density (float | None):
            Fraction of non-zero entries in the coefficient matrix.
        density_type (DensityType | None):
            Classification of the density (e.g., sparse, dense).
    """

    def __init__(
        self,
        coefficients: dict | ArrayLike | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a QUBOInstance.

        Args:
            coefficients (dict | ArrayLike | None):
                Coefficients of the QUBO problem. Can be a dictionary, array-like object, or None.
            device (str):
                Device where tensors are allocated (default: "cpu").
            dtype (torch.dtype):
                Data type of the tensors (default: torch.float32).
        """
        self.device = device
        self.dtype = dtype
        self.size: int | None
        self._coefficients: torch.Tensor | None = None
        self.solution: QUBOSolution | None = None
        self.density: float | None = None
        self.density_type: DensityType | None = None

        if coefficients is not None:
            self.coefficients = coefficients

    @property
    def coefficients(self) -> torch.Tensor:
        """
        Getter for the QUBO coefficient matrix.

        Returns:
            torch.Tensor:
                Tensor of shape (size, size) representing the QUBO coefficients.
        """
        assert self._coefficients is not None
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coeffs: dict | ArrayLike) -> None:
        """
        Setter for the QUBO coefficient matrix, with validation checks.

        Checks:
            All off-diagonal coefficients must be non-negative.

        Exits the program with an error message if a check fails.
        """
        # Convert input to tensor
        tensor = convert_to_tensor(coeffs, device=self.device, dtype=self.dtype)
        size = tensor.shape[0]

        # Off-diagonal negativity check
        off_diag = tensor.triu(diagonal=1)
        if (off_diag < 0).any():
            raise ValueError(
                "Error: Negative off-diagonal coefficient detected. "
                "All off-diagonal coefficients must be >= 0."
            )

        # All checks passed, assign the tensor and update metrics
        self._coefficients = tensor
        self.size = size
        self.update_metrics()

    def set_coefficients(
        self, new_coefficients: dict[tuple[int, int], float] | None = None
    ) -> None:
        """
        Updates the coefficients of the QUBO problem.

        Args:
            new_coefficients (dict[tuple[int, int], float] | None):
                Dictionary of new coefficients to set. Keys are (row, column) tuples.
        """
        if not new_coefficients:
            return

        max_index = max(max(i, j) for i, j in new_coefficients.keys())
        if self.size and max_index >= self.size:
            self._expand_size(max_index + 1)

        indices = torch.tensor(list(new_coefficients.keys()), dtype=torch.long, device=self.device)
        values = torch.tensor(list(new_coefficients.values()), dtype=self.dtype, device=self.device)
        self._coefficients[indices[:, 0], indices[:, 1]] = values  # type: ignore[index]
        off_diagonal_mask = indices[:, 0] != indices[:, 1]
        symmetric_indices = indices[off_diagonal_mask]
        self._coefficients[symmetric_indices[:, 1], symmetric_indices[:, 0]] = values[  # type: ignore[index]
            off_diagonal_mask
        ]

        self.update_metrics()

    def _expand_size(self, new_size: int) -> None:
        """
        Expands the size of the coefficient matrix to accommodate larger indices.

        Args:
            new_size (int):
                New size of the coefficient matrix.
        """
        if self._coefficients is not None:
            expanded_coefficients = torch.zeros(
                (new_size, new_size), dtype=self.dtype, device=self.device
            )
            expanded_coefficients[: self.size, : self.size] = self._coefficients
            self._coefficients = expanded_coefficients
        self.size = new_size

    def update_metrics(self) -> None:
        """
        Updates the density metrics of the QUBO problem.
        """
        if self._coefficients is not None:
            self.density = calculate_density(self._coefficients, self.size)
            self.density_type = classify_density(self.density)
        else:
            self.density = self.density_type = None

    def evaluate_solution(self, solution: list | tuple | ArrayLike) -> float:
        """
        Evaluates a solution for the QUBO problem.

        Args:
            solution (list | tuple | ArrayLike):
                Solution vector to evaluate.

        Returns:
            float:
                The cost of the given solution.

        Raises:
            ValueError: If the solution size does not match the QUBO size.
        """
        solution_tensor = convert_to_tensor(solution, device=self.device, dtype=self.dtype)
        if self._coefficients is None or solution_tensor.size(0) != self.size:
            raise ValueError("Solution size does not match the QUBO problem size.")
        cost = torch.matmul(
            solution_tensor, torch.matmul(self._coefficients, solution_tensor)
        ).item()
        solution = solution_tensor
        return float(cost)

    def __repr__(self) -> str:
        """
        Returns a string representation of the QUBOInstance.

        Returns:
            str: A dictionary-like string summarizing the instance.
        """
        return repr(
            f"QUBOInstance of size = {self.size},"
            f"density = {round(self.density, 2) if self.density else None},"
        )


# Modules to be automatically added to the qubosolver namespace
__all__ = ["QUBOInstance"]  # type: ignore
