from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from qubosolver.data_utils import generate_symmetric_mask
from qubosolver.qubo_types import SolutionStatusType

if TYPE_CHECKING:
    pass

# Modules to be automatically added to the qubosolver namespace
__all__ = ["QUBOSolution", "QUBODataset"]  # type: ignore


@dataclass
class QUBOSolution:
    """
    Represents a solution to a QUBO problem.

    Attributes:
        bitstrings (torch.Tensor):
            Tensor of shape (num_solutions, bitstring_length), containing the bitstring solutions.
            Each entry is an integer tensor with 0s and 1s.
        counts (torch.Tensor | None):
            Tensor of shape (num_solutions,), containing the count of occurrences of each bitstring.
            Optional, can be None.
        probabilities (torch.Tensor | None):
            Tensor of shape (num_solutions,), containing the probability of each bitstring solution.
            Optional, can be None.
        costs (torch.Tensor):
            Tensor of shape (num_solutions,), containing the cost associated with each
            bitstring solution.
    """

    bitstrings: torch.Tensor
    costs: torch.Tensor
    counts: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    solution_status: SolutionStatusType = SolutionStatusType.UNPROCESSED

    def compute_costs(self, instance: Any) -> torch.Tensor:
        """
        Computes the cost for each bitstring solution based on the provided QUBO instance.

        Args:
            instance (QUBOInstance): The QUBO instance containing the QUBO matrix.

        Returns:
            torch.Tensor: A tensor of costs for each bitstring.
        """
        # Retrieve the QUBO matrix from the QUBOInstance
        QUBO = instance.coefficients  # Assuming `coefficients` holds the QUBO matrix

        costs = []
        for bitstring in self.bitstrings:
            if isinstance(bitstring, str):
                z = torch.tensor([int(b) for b in bitstring], dtype=torch.float32)
            elif isinstance(bitstring, torch.Tensor):
                z = bitstring.detach().clone()
            else:
                z = torch.tensor(bitstring, dtype=torch.float32)
            cost = torch.matmul(
                z.permute(*torch.arange(z.ndim - 1, -1, -1)), torch.matmul(QUBO, z)
            ).item()  # Use the QUBO matrix from the instance
            costs.append(cost)

        return torch.tensor(costs)

    def compute_probabilities(self) -> torch.Tensor:
        """
        Computes the probabilities of each bitstring solution based on their counts.

        Returns:
            torch.Tensor: A tensor of probabilities for each bitstring.
        """
        if self.counts is None:
            raise ValueError("Counts are required to compute probabilities.")

        total_counts = self.counts.sum().item()
        probabilities = (
            self.counts / total_counts if total_counts > 0 else torch.zeros_like(self.counts)
        )
        return probabilities

    def sort_by_cost(self) -> None:
        """
        Sorts the QUBOSolution in-place by increasing cost.

        Reorders bitstrings, costs, counts, and probabilities (if available)
        based on the ascending order of the costs.
        """

        sorted_indices = torch.argsort(self.costs)
        self.bitstrings = self.bitstrings[sorted_indices]
        self.costs = self.costs[sorted_indices]
        if self.counts is not None:
            self.counts = self.counts[sorted_indices]
        if self.probabilities is not None:
            self.probabilities = self.probabilities[sorted_indices]


class QUBODataset(Dataset):
    """
    Represents a dataset for Quadratic Unconstrained Binary Optimization (QUBO) problems.

    Attributes:
        coefficients (torch.Tensor):
            Tensor of shape (size, size, num_instances), containing the QUBO coefficient matrices.
        solutions (list[QUBOSolution] | None):
            Optional list of QUBOSolution objects corresponding to each instance in the dataset.

    Methods:
        __len__():
            Returns the number of instances in the dataset.
        __getitem__(idx):
            Retrieves the coefficient matrix and optionally the solution for the
            specified index.
        from_random():
            Class method to generate a QUBODataset with random coefficient matrices.
    """

    def __init__(self, coefficients: torch.Tensor, solutions: list[QUBOSolution] | None = None):
        """
        Initializes a QUBODataset.

        Args:
            coefficients (torch.Tensor):
                Tensor of shape (size, size, num_instances), containing the QUBO
                coefficient matrices.
            solutions (list[QUBOSolution] | None):
                Optional list of QUBOSolution objects corresponding to each instance
                in the dataset.
        """
        self.coefficients = coefficients
        self.solutions = solutions

    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.

        Returns:
            int: The number of coefficient matrices (num_instances).
        """
        return int(self.coefficients.shape[2])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, QUBOSolution | None]:
        """
        Retrieves the coefficient matrix and optionally the solution for the specified index.

        Args:
            idx (int):
                Index of the dataset instance to retrieve.

        Returns:
            tuple[torch.Tensor, QUBOSolution | None]:
                The coefficient matrix of shape (size, size) and optionally
                the corresponding QUBOSolution.
        """
        if self.solutions is not None:
            return self.coefficients[:, :, idx], self.solutions[idx]
        return self.coefficients[:, :, idx], None

    @classmethod
    def from_random(
        cls,
        n_matrices: int,
        matrix_dim: int,
        densities: list[float] = [0.5],
        coefficient_bounds: tuple[float, float] = (-10.0, 10.0),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> QUBODataset:
        """
        Generates a QUBODataset with random QUBO coefficient matrices.

        Generation Steps:
        1. Initialize a reproducible random generator.
        2. Create a storage tensor for coefficients.
        3. For each density:
            a. Compute the exact target number of non-zero elements.
            b. For each instance:
                i.  Generate a symmetric boolean mask with an exact number of True elements.
                ii. Generate random values within the coefficient_bounds.
                iii. Apply the mask to zero out unselected elements.
                iv. Symmetrize the matrix by mirroring the upper triangle onto the lower triangle.
                v. Force all off-diagonal coefficients to be positive.
                vi. Ensure that at least one diagonal element is negative.
                vii. Ensure at least one coefficient equals the upper bound, excluding
                any diagonal already at the lower bound.
        4. Return a QUBODataset instance containing the generated matrices.

        Args:
            n_matrices (int): Number of QUBO matrices to generate for each density.
            matrix_dim (int): The dimension of each QUBO matrix.
            densities (list[float], optional): List of densities (ratio of non-zero elements).
                Defaults to [0.5].
            coefficient_bounds (tuple[float, float], optional): Range (min, max) of
                random values for the coefficients. Defaults to (-10.0, 10.0).
            device (str): Device for the tensors. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type for the coefficient tensors.
                Defaults to torch.float32.
            seed (int | None, optional): Seed for reproducibility. Defaults to None.

        Returns:
            QUBODataset: A dataset containing the generated coefficient matrices.
        """
        # Step 1: Initialize a reproducible random generator.
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        # Step 2: Create a tensor for the coefficients.
        total_instances = n_matrices * len(densities)
        coefficients = torch.zeros(
            matrix_dim, matrix_dim, total_instances, device=device, dtype=dtype
        )

        # Step 3: Generate matrices for each density.
        idx = 0
        for d in densities:
            target = int(d * matrix_dim * matrix_dim)
            for _ in range(n_matrices):
                mask = generate_symmetric_mask(matrix_dim, target, device, generator)
                random_vals = torch.empty(
                    matrix_dim, matrix_dim, device=device, dtype=dtype
                ).uniform_(*coefficient_bounds, generator=generator)
                random_vals = random_vals * mask.to(dtype)

                original_diag = random_vals.diag().clone()
                coeff = torch.triu(random_vals, diagonal=1)
                coeff = coeff + coeff.T
                coeff.diagonal().copy_(original_diag)

                off_diag = ~torch.eye(matrix_dim, dtype=torch.bool, device=device)
                coeff[off_diag] = coeff[off_diag].abs()

                if not (coeff.diag() < 0).any():
                    diag_vals = coeff.diag()
                    non_neg = (diag_vals >= 0).nonzero(as_tuple=True)[0]
                    diag_idx = (
                        non_neg[0].item()
                        if non_neg.numel() > 0
                        else torch.randint(
                            0, matrix_dim, (1,), device=device, generator=generator
                        ).item()
                    )
                    if coefficient_bounds[0] < 0:
                        neg_val = coefficient_bounds[0]
                    else:
                        neg_val = (
                            -torch.empty(1, device=device, dtype=dtype)
                            .uniform_(*coefficient_bounds, generator=generator)
                            .abs()
                            .item()
                        )
                    coeff[diag_idx, diag_idx] = neg_val

                if not (coeff == coefficient_bounds[1]).any():
                    nz = (coeff != 0).nonzero(as_tuple=False)
                    filtered = [
                        idx_pair
                        for idx_pair in nz.tolist()
                        if not (
                            idx_pair[0] == idx_pair[1]
                            and coeff[idx_pair[0], idx_pair[1]].item() == coefficient_bounds[0]
                        )
                    ]
                    if filtered:
                        chosen = filtered[
                            torch.randint(
                                0,
                                len(filtered),
                                (1,),
                                device=device,
                                generator=generator,
                            ).item()
                        ]
                    else:
                        chosen = [
                            torch.randint(
                                0, matrix_dim, (1,), device=device, generator=generator
                            ).item()
                        ] * 2
                    i_ch, j_ch = chosen
                    coeff[i_ch, j_ch] = coefficient_bounds[1]
                    if i_ch != j_ch:
                        coeff[j_ch, i_ch] = coefficient_bounds[1]

                coefficients[:, :, idx] = coeff
                idx += 1

        # Step 4: Return the dataset.
        return cls(coefficients=coefficients)
