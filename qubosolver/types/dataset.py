# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Dataset container for batches of QUBO problem instances.

Provides [`Dataset`][qubosolver.Dataset], a dataset of QUBO
coefficient matrices paired with optional ground-truth solutions, along with
random dataset generation and binary (de)serialization helpers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from qubosolver._io import utils as io_utils

from .instance import Instance
from .random import numpy_rng
from .solution import Solution

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

    from qubosolver._io.utils import FileLike


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
        self,
        matrices: NDArray[np.float64],
        solutions: Sequence[Solution] = (),
        *,
        copy: bool = True,
    ) -> None:
        """Build a dataset from `matrices` and optional `solutions`, deep-copying by default."""
        if copy:
            matrices = matrices.copy()
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
    def from_random(  # noqa: PLR0913, PLR0915
        cls,
        n_matrices: int,
        matrix_dim: int,
        *,
        densities: Sequence[float] = (0.5,),
        coefficient_bounds: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
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
        rng = rng or numpy_rng()

        # Step 2: Create an array for the coefficients.
        total_instances = n_matrices * len(densities)
        coefficients = np.zeros((matrix_dim, matrix_dim, total_instances))

        # Step 3: Generate matrices for each density.
        idx = 0
        for d in densities:
            target = int(d * matrix_dim * matrix_dim)
            for idx in range(n_matrices):
                # generate mask
                mask = _generate_symmetric_mask(matrix_dim, target, rng)

                # random sampling and apply mask
                random_vals = rng.uniform(*coefficient_bounds, size=(matrix_dim, matrix_dim))
                random_vals = random_vals * mask

                original_diag = random_vals.diagonal().copy()
                coeff = np.triu(random_vals, k=1)
                coeff = coeff + coeff.T
                np.fill_diagonal(coeff, original_diag)

                off_diag = ~np.eye(matrix_dim, dtype=bool)
                coeff[off_diag] = np.abs(coeff[off_diag])
                if negative_offdiag_rate > 0.0:
                    # make non-diagonal negative elements
                    rate = float(max(0.0, min(1.0, negative_offdiag_rate)))
                    upper_mask = np.triu(mask, k=1)
                    nz_pairs = np.argwhere(upper_mask)
                    M = nz_pairs.shape[0]
                    # Return K negative elements
                    if M > 0:
                        K = max(1, round(rate * M))
                        perm = rng.permutation(M)[:K]
                        chosen = nz_pairs[perm]
                        i_idx, j_idx = chosen[:, 0], chosen[:, 1]
                        vals = coeff[i_idx, j_idx]
                        neg_vals = -vals
                        coeff[i_idx, j_idx] = neg_vals
                        coeff[j_idx, i_idx] = neg_vals
                    else:
                        # Edge case to force creating one a negative element
                        i, j = 0, 1 if matrix_dim > 1 else (0, 0)
                        coeff[i, j] = -rng.random() * abs(coefficient_bounds[0])
                        coeff[j, i] = coeff[i, j]
                if not (coeff.diagonal() < 0).any():
                    diag_vals = coeff.diagonal()
                    non_neg = np.flatnonzero(diag_vals >= 0)
                    diag_idx = (
                        int(non_neg[0]) if non_neg.size > 0 else int(rng.integers(0, matrix_dim))
                    )
                    if coefficient_bounds[0] < 0:
                        neg_val = coefficient_bounds[0]
                    else:
                        neg_val = -abs(rng.uniform(*coefficient_bounds))
                    coeff[diag_idx, diag_idx] = neg_val
                if not (coeff == coefficient_bounds[1]).any():
                    # do not select negative coefficients
                    nz = np.argwhere(coeff > 0)
                    filtered = [
                        idx_pair
                        for idx_pair in nz.tolist()
                        if not (
                            idx_pair[0] == idx_pair[1]
                            and coeff[idx_pair[0], idx_pair[1]] == coefficient_bounds[0]
                        )
                    ]
                    if filtered:
                        i_ch, j_ch = filtered[int(rng.integers(0, len(filtered)))]
                    else:
                        i_ch = j_ch = int(rng.integers(0, matrix_dim))
                    coeff[i_ch, j_ch] = coefficient_bounds[1]
                    if i_ch != j_ch:
                        coeff[j_ch, i_ch] = coefficient_bounds[1]

                coefficients[:, :, idx] = coeff

        # Step 4: Return the dataset.
        return cls(matrices=coefficients, copy=False)

    def save(self, file_like: FileLike[bytes]) -> None:
        """Persist this dataset to disk.

        Args:
            file_like: Destination file path or writable binary file object.

        Example:
            ```python
            from pathlib import Path

            with Path("dataset.bin").open("wb") as f:
                dataset.save(f)
            ```
        """
        with io_utils.open(file_like, "wb") as f:
            io_utils.save_header(f)
            io_utils.save_array(f, self.matrices)
            io_utils.save(f, ">I", len(self.solutions))
            # Written into the already-open stream *f*, not into `file_like`:
            # re-opening a path here would truncate everything written above.
            for s in self.solutions:
                s.save(f)

    @staticmethod
    def load(file_like: FileLike[bytes]) -> Dataset:
        """Load a dataset previously saved with [`save`][].

        Args:
            file_like: Source file path or readable binary file object,
                as produced by [`save`][].

        Returns:
            The deserialized dataset, including solutions if they were present when the
                file was saved.

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
            matrices = io_utils.load_array(f)
            n = io_utils.load(f, ">I")
            solutions = [Solution.load(f) for _ in range(n)]

        return Dataset(matrices, solutions, copy=False)


def _generate_symmetric_mask(size: int, target: int, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Generate a symmetric boolean mask with exactly *target* ``True`` entries.

    The mask is used by :meth:`Dataset.from_random` to enforce an exact
    sparsity level.  Symmetry is maintained by setting both ``mask[i, j]`` and
    ``mask[j, i]`` whenever an off-diagonal position is selected, so the
    effective number of *unique* off-diagonal pairs is ``(target - x) // 2``
    where ``x`` is the number of selected diagonal entries.

    Args:
        size: Side length of the square mask (``size x size``).
        target: Exact number of ``True`` entries in the returned mask.
        rng: Random number generator for reproducible sampling.

    Returns:
        Boolean array of shape ``(size, size)`` with exactly *target*
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
        x = possible_x[int(rng.integers(0, len(possible_x)))]
        y = (target - x) // 2

    mask = np.zeros((size, size), dtype=bool)
    diag_indices = rng.permutation(size)[:x]
    for i in diag_indices.tolist():
        mask[i, i] = True

    upper_indices = np.array(
        [(i, j) for i in range(size) for j in range(i + 1, size)], dtype=np.int64
    ).reshape(-1, 2)
    if upper_indices.shape[0] > 0 and y > 0:
        perm = rng.permutation(upper_indices.shape[0])[:y]
        chosen_upper = upper_indices[perm]
        for i, j in chosen_upper.tolist():
            mask[i, j] = True
            mask[j, i] = True
    return mask
