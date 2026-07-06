from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import Dataset
from pathlib import Path

from .solution import QUBOSolution
from . import matrix
from .random import torch_rng


class QUBODataset(Dataset):
    """A PyTorch ``Dataset`` of QUBO (Quadratic Unconstrained Binary Optimization) instances.

    Each instance is represented by a square coefficient matrix ``Q`` such that the
    optimization objective is ``x^T Q x``, where ``x`` is a binary vector.

    Attributes:
        matrix (torch.Tensor):
            Coefficient matrices stored as a 3-D tensor of shape
            ``(size, size, num_instances)``.  The third axis indexes individual
            problem instances.
        solutions (list[QUBOSolution]):
            Known solutions for each instance.  Empty when the dataset was
            created without ground-truth solutions (e.g. via :meth:`from_random`).
    """

    def __init__(self, coefficients: torch.Tensor, solutions: list[QUBOSolution] = []):
        """
        Args:
            coefficients (torch.Tensor):
                Coefficient matrices of shape ``(size, size, num_instances)``.
                ``coefficients[:, :, i]`` is the ``i``-th QUBO matrix.
            solutions (list[QUBOSolution]):
                Ground-truth solutions, one per instance.  Pass an empty list
                (default) when solutions are unknown.

        Note:
            ``coefficients`` is stored directly (no copy).  Mutating the tensor
            after construction will affect the dataset.
        """
        self.matrix = coefficients
        self.solutions = solutions or []

    def __len__(self) -> int:
        """Return the number of QUBO instances in the dataset (``num_instances``)."""
        return int(self.matrix.shape[2])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, QUBOSolution]:
        """Return the coefficient matrix and solution for instance *idx*.

        Args:
            idx (int): Zero-based index of the instance.

        Returns:
            tuple[torch.Tensor, QUBOSolution]:
                ``(Q, solution)`` where ``Q`` has shape ``(size, size)``.
                When no solutions were provided, ``solution`` is an empty
                :class:`~.QUBOSolution`.
        """
        if self.solutions:
            return self.matrix[:, :, idx], self.solutions[idx]
        return self.matrix[:, :, idx], QUBOSolution()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, QUBOSolution]]:
        """Iterate over all ``(coefficient_matrix, solution)`` pairs in order.

        Yields:
            tuple[torch.Tensor, QUBOSolution]:
                Same as :meth:`__getitem__` for each index ``0 … len(self)-1``.
        """
        return map(self.__getitem__, range(len(self)))

    @classmethod
    def from_random(
        cls,
        n_matrices: int,
        matrix_dim: int,
        *,
        densities: list[float] = [0.5],
        coefficient_bounds: tuple[float, float] = (-10.0, 10.0),
        dtype: torch.dtype = matrix.dtype(),
        rng: torch.Generator = torch_rng(),
        negative_offdiag_rate: float = 0.0,
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
            dtype (torch.dtype, optional): Data type for the coefficient tensors.
                Defaults to torch.float32.
            negative_offdiag_rate (float, optional): off-diagonal negative coefficients rate.
                Defaults to None, meaning no off-diagonal coefficient will be present.

        Returns:
            QUBODataset: A dataset containing the generated coefficient matrices.
        """
        # Step 1: Initialize a reproducible random generator.
        device = rng.device.type

        # Step 2: Create a tensor for the coefficients.
        total_instances = n_matrices * len(densities)
        coefficients = torch.zeros(
            matrix_dim, matrix_dim, total_instances, device=device, dtype=dtype
        )

        # Step 3: Generate matrices for each density.
        idx = 0
        for d in densities:
            target = int(d * matrix_dim * matrix_dim)
            for idx in range(n_matrices):

                # generate mask
                mask = _generate_symmetric_mask(matrix_dim, target, device, rng)

                # random sampling and apply mask
                random_vals = torch.empty(
                    matrix_dim, matrix_dim, device=device, dtype=dtype
                ).uniform_(*coefficient_bounds, generator=rng)
                random_vals = random_vals * mask.to(dtype)

                original_diag = random_vals.diag().clone()
                coeff = torch.triu(random_vals, diagonal=1)
                coeff = coeff + coeff.T
                coeff.diagonal().copy_(original_diag)

                off_diag = ~torch.eye(matrix_dim, dtype=torch.bool, device=device)
                coeff[off_diag] = coeff[off_diag].abs()
                if negative_offdiag_rate > 0.0:
                    # make non-diagonal negative elements
                    rate = float(max(0.0, min(1.0, negative_offdiag_rate)))
                    upper_mask = torch.triu(mask, diagonal=1)
                    nz_pairs = upper_mask.nonzero(as_tuple=False)
                    M = nz_pairs.size(0)
                    # Return K negative elements
                    if M > 0:
                        K = max(1, int(round(rate * M)))
                        perm = torch.randperm(M, generator=rng, device=device)[:K]
                        chosen = nz_pairs[perm]
                        i_idx, j_idx = chosen[:, 0], chosen[:, 1]
                        vals = coeff[i_idx, j_idx]
                        neg_vals = -vals
                        coeff[i_idx, j_idx] = neg_vals
                        coeff[j_idx, i_idx] = neg_vals
                    else:
                        # Edge case to force creating one a negative element
                        i, j = 0, 1 if matrix_dim > 1 else (0, 0)
                        coeff[i, j] = -torch.rand(1, device=device, generator=rng) * abs(
                            coefficient_bounds[0]
                        )
                        coeff[j, i] = coeff[i, j]
                if not (coeff.diag() < 0).any():
                    diag_vals = coeff.diag()
                    non_neg = (diag_vals >= 0).nonzero(as_tuple=True)[0]
                    diag_idx = (
                        int(non_neg[0].item())
                        if non_neg.numel() > 0
                        else int(
                            torch.randint(0, matrix_dim, (1,), device=device, generator=rng).item()
                        )
                    )
                    if coefficient_bounds[0] < 0:
                        neg_val = coefficient_bounds[0]
                    else:
                        neg_val = (
                            -torch.empty(1, device=device, dtype=dtype)
                            .uniform_(*coefficient_bounds, generator=rng)
                            .abs()
                            .item()
                        )
                    coeff[diag_idx, diag_idx] = neg_val
                if not (coeff == coefficient_bounds[1]).any():
                    # do not select negative coefficients
                    nz = (coeff > 0).nonzero(as_tuple=False)
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
                            int(
                                torch.randint(
                                    0,
                                    len(filtered),
                                    (1,),
                                    device=device,
                                    generator=rng,
                                    dtype=torch.int64,
                                ).item()
                            )
                        ]
                    else:
                        chosen = torch.randint(
                            0, matrix_dim, (1,), device=device, generator=rng
                        ).repeat(2)
                    i_ch, j_ch = chosen
                    coeff[i_ch, j_ch] = coefficient_bounds[1]
                    if i_ch != j_ch:
                        coeff[j_ch, i_ch] = coefficient_bounds[1]

                coefficients[:, :, idx] = coeff

        # Step 4: Return the dataset.
        return cls(coefficients=coefficients)

    @staticmethod
    def save(dataset: QUBODataset, filepath: Path) -> None:
        """Persist a dataset to disk using `torch.save`.

        Args:
            dataset (QUBODataset): The dataset to serialise.
            filepath (Path): Destination file path.  The file is created or
                overwritten.  Use a ``.pt`` or ``.pth`` extension by convention.
        """
        return _save_qubo_dataset(dataset, filepath)

    @staticmethod
    def load(filepath: Path) -> QUBODataset:
        """Load a dataset previously saved with :meth:`save`.

        Args:
            filepath (Path): Path to the file produced by :meth:`save`.

        Returns:
            QUBODataset: The deserialised dataset, including solutions if they
            were present when the file was saved.
        """
        return _load_qubo_dataset(filepath)


def _save_qubo_dataset(dataset: QUBODataset, filepath: Path) -> None:
    """Serialise *dataset* to *filepath* via `torch.save`.

    The file stores a dict with two keys:

    * ``"coefficients"`` — the raw ``(size, size, num_instances)`` tensor.
    * ``"solutions"`` — ``None`` when no solutions exist, otherwise a list of
      dicts each containing ``bitstrings``, ``counts``, ``probabilities``, and
      ``costs`` as returned by the corresponding :class:`~.QUBOSolution`
      attributes.

    Args:
        dataset (QUBODataset): Dataset to serialise.
        filepath (Path): Destination file path (created or overwritten).
    """
    data: dict[str, Any] = {"coefficients": dataset.matrix, "solutions": None}
    if dataset.solutions is not None:
        data["solutions"] = [
            {
                "bitstrings": solution.bitstrings,
                "counts": solution.counts,
                "probabilities": solution.probabilities,
                "costs": solution.costs,
            }
            for solution in dataset.solutions
        ]
    torch.save(data, filepath)


def _load_qubo_dataset(filepath: Path) -> QUBODataset:
    """Deserialise a :class:`QUBODataset` from *filepath*.

    The file must have been produced by `_save_qubo_dataset`.

    Args:
        filepath (Path): Path to the serialised dataset file.

    Returns:
        QUBODataset: Restored dataset, with solutions populated when they were
        present in the file.
    """
    data = torch.load(filepath)
    solutions = []
    if data["solutions"] is not None:
        solutions = [
            QUBOSolution(
                bitstrings=solution["bitstrings"],
                counts=solution["counts"],
                probabilities=solution["probabilities"],
                costs=solution["costs"],
            )
            for solution in data["solutions"]
        ]
    return QUBODataset(coefficients=data["coefficients"], solutions=solutions)


def _generate_symmetric_mask(
    size: int, target: int, device: str, rng: torch.Generator
) -> torch.Tensor:
    """Generate a symmetric boolean mask with exactly *target* ``True`` entries.

    The mask is used by :meth:`QUBODataset.from_random` to enforce an exact
    sparsity level.  Symmetry is maintained by setting both ``mask[i, j]`` and
    ``mask[j, i]`` whenever an off-diagonal position is selected, so the
    effective number of *unique* off-diagonal pairs is ``(target - x) // 2``
    where ``x`` is the number of selected diagonal entries.

    Args:
        size (int): Side length of the square mask (``size × size``).
        target (int): Exact number of ``True`` entries in the returned mask.
        device (str): Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        rng (torch.Generator): Random number generator for reproducible sampling.

    Returns:
        torch.Tensor: Boolean tensor of shape ``(size, size)`` with exactly
        *target* ``True`` values and perfect symmetry (``mask == mask.T``).
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
            int(torch.randint(0, len(possible_x), (1,), device=device, generator=rng).item())
        ]
        y = (target - x) // 2

    mask = torch.zeros((size, size), dtype=torch.bool, device=device)
    diag_indices = torch.randperm(size, device=device, generator=rng)[:x]
    for i in diag_indices.tolist():
        mask[i, i] = True

    upper_indices = torch.tensor(
        [(i, j) for i in range(size) for j in range(i + 1, size)],
        device=device,
    )
    if upper_indices.size(0) > 0 and y > 0:
        perm = torch.randperm(upper_indices.size(0), device=device, generator=rng)[:y]
        chosen_upper = upper_indices[perm]
        for i, j in chosen_upper.tolist():
            mask[i, j] = True
            mask[j, i] = True
    return mask
