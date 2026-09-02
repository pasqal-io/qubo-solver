"""Dataset container for batches of QUBO problem instances.

Provides [`Dataset`][qubosolver.Dataset], a dataset of QUBO
coefficient matrices paired with optional ground-truth solutions, along with
random dataset generation and binary (de)serialization helpers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import torch
import io

from .solution import Solution
from .instance import Instance
from . import matrix
from .random import torch_rng
from qubosolver._io import utils as io_utils


class Dataset:
    """A dataset of QUBO instances.

    Each instance is represented by a square matrix $Q$ such that the
    optimization objective is $x^T Q x$, where $x$ is a binary vector.

    Args:
        matrices: Matrices of shape ``(size, size, num_instances)``.
            ``matrices[:, :, i]`` is the ``i``-th QUBO matrix.
        solutions: Ground-truth solutions, one per instance.  Pass an empty list
            (default) when solutions are unknown.
        copy: Whether to deep-copy ``matrices`` and ``solutions`` on
            construction.  Pass ``False`` to store the
            given values directly (no copy), e.g. when the caller already
            owns them exclusively.

    Attributes:
        matrices: Matrices stored as a 3-D tensor of shape
            ``(size, size, num_instances)``.  The third axis indexes individual
            problem instances.
        solutions: Known solutions for each instance.  Empty when the dataset was
            created without ground-truth solutions (e.g. via [`from_random`][]).

    Note:
        ``matrices`` and ``solutions`` are deep-copied on construction by
        default.  Mutating the values passed in afterwards will not affect
        the dataset, unless ``copy=False`` was given.
    """

    def __init__(
        self, matrices: torch.Tensor, solutions: list[Solution] = [], *, copy: bool = True
    ):
        if copy:
            matrices = matrices.detach().clone()
            solutions = deepcopy(solutions)
        self.matrices = matrices
        self.solutions = solutions

    def __len__(self) -> int:
        """Return the number of QUBO instances in the dataset."""
        return int(self.matrices.shape[2])

    def __getitem__(self, idx: int) -> tuple[Instance, Solution]:
        """Return the matrix and solution for instance *idx*.

        Args:
            idx: Zero-based index of the instance.

        Returns:
            A symmetric matrix and a solution ``(Q, solution)``.
                When no solutions were provided, ``solution`` is an empty
                [`Solution`][].
        """
        instance = Instance(self.matrices[:, :, idx])
        if self.solutions:
            return instance, self.solutions[idx]
        return instance, Solution()

    def __iter__(self) -> Iterator[tuple[Instance, Solution]]:
        """Iterate over all ``(matrix, solution)`` pairs in order.

        Yields:
            A matrix and a solution.
                Same as [`__getitem__`][] for each index ``0 … len(self)-1``.
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
    ) -> Dataset:
        """Generates a Dataset of random, symmetric QUBO coefficient matrices.

        For each requested density, `n_matrices` symmetric matrices of shape
        ``(matrix_dim, matrix_dim)`` are generated with (approximately) that
        fraction of non-zero entries. Off-diagonal coefficients are positive
        (unless flipped negative by `negative_offdiag_rate`), each matrix is
        guaranteed at least one negative diagonal element and at least one
        coefficient equal to `coefficient_bounds[1]`, so that the resulting
        instances are non-trivial to solve.

        Args:
            n_matrices: Number of QUBO matrices to generate for each density.
            matrix_dim: The dimension of each QUBO matrix.
            densities: List of densities (ratio of non-zero elements).
            coefficient_bounds: Range (min, max) of
                random values for the coefficients.
            dtype: Data type for the coefficient matrices.
            rng: Random number generator controlling
                the sampling.
            negative_offdiag_rate: Fraction of the non-zero
                off-diagonal coefficients to flip negative.
                A value of 0 means that no off-diagonal coefficient is negative.

        Returns:
            A dataset containing ``n_matrices * len(densities)`` generated
                coefficient matrices, with no associated solutions.
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
        return cls(matrices=coefficients, copy=False)

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], dataset: Dataset) -> None:
        """Persist a dataset to disk using [`torch.save`][].

        Args:
            file_like: Destination file path or writable binary file object.
            dataset: The dataset to serialize.

        Example:
            ```python
            from pathlib import Path

            with Path("dataset.bin").open("wb") as f:
                Dataset.save(f, dataset)
            ```
        """
        with io_utils.open(file_like, "wb") as f:
            io_utils.save_header(f)
            buffer = io.BytesIO()
            torch.save(dataset.matrices, buffer)
            io_utils.save_sized_buffer(f, buffer.getbuffer())
            io_utils.save(f, ">I", len(dataset.solutions))
            # Written into the already-open stream *f*, not into `file_like`:
            # re-opening a path here would truncate everything written above.
            for s in dataset.solutions:
                Solution.save(f, s)

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> Dataset:
        """Load a dataset previously saved with [`save`][].

        Args:
            file_like: Source file path or readable binary file object,
                as produced by [`save`][].

        Returns:
            The deserialized dataset, including solutions if they were present when the file was saved.

        Raises:
            ValueError: If the stream is not a qubosolver file.

        Example:
            ```python
            from pathlib import Path

            with Path("dataset.bin").open("rb") as f:
                dataset = Dataset.load(f)
            ```
        """
        with io_utils.open(file_like, "rb") as f:
            io_utils.load_header(f)
            # torch.load might consume too much of the src buffer.
            #  Use a dedicated limited buffer
            buffer = io.BytesIO(io_utils.load_sized_buffer(f))
            matrices = torch.load(buffer, weights_only=True)
            n = io_utils.load(f, ">I")
            solutions = [Solution.load(f) for _ in range(n)]

        return Dataset(matrices, solutions, copy=False)


def _generate_symmetric_mask(
    size: int, target: int, device: str, rng: torch.Generator
) -> torch.Tensor:
    """Generate a symmetric boolean mask with exactly *target* ``True`` entries.

    The mask is used by :meth:`Dataset.from_random` to enforce an exact
    sparsity level.  Symmetry is maintained by setting both ``mask[i, j]`` and
    ``mask[j, i]`` whenever an off-diagonal position is selected, so the
    effective number of *unique* off-diagonal pairs is ``(target - x) // 2``
    where ``x`` is the number of selected diagonal entries.

    Args:
        size: Side length of the square mask (``size × size``).
        target: Exact number of ``True`` entries in the returned mask.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        rng: Random number generator for reproducible sampling.

    Returns:
        Boolean tensor of shape ``(size, size)`` with exactly *target*
            ``True`` values and perfect symmetry (``mask == mask.T``).
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
