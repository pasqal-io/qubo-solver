# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check

from qubosolver import Dataset, numpy_rng
from qubosolver.types.instance import _calculate_density
from qubosolver.types.solution import Solution

if TYPE_CHECKING:
    from pathlib import Path


def test_dataset_copies_input_matrices() -> None:
    """Test that Dataset copies the input array instead of aliasing it."""
    matrices = np.zeros((3, 3, 2))
    dataset = Dataset(matrices)

    matrices[0, 0, 0] = 42.0

    check.equal(dataset.matrices[0, 0, 0].item(), 0.0)


def test_dataset_copies_input_solutions() -> None:
    """Test that Dataset deep-copies the input solutions instead of aliasing them."""
    matrices = np.zeros((3, 3, 1))
    solution = Solution(
        bitstrings=np.zeros((1, 3), dtype=np.int8),
        costs=np.zeros(1),
        counts=np.zeros(1, dtype=np.int64),
        probabilities=np.zeros(1),
    )
    dataset = Dataset(matrices, [solution])

    solution.bitstrings[0, 0] = 1
    solution.costs[0] = 42.0

    check.equal(dataset.solutions[0].bitstrings[0, 0].item(), 0)
    check.equal(dataset.solutions[0].costs[0].item(), 0.0)


def test_dataset_copy_false_aliases_input() -> None:
    """Test that Dataset(copy=False) stores the given matrices/solutions directly."""
    matrices = np.zeros((3, 3, 1))
    solution = Solution(
        bitstrings=np.zeros((1, 3), dtype=np.int8),
        costs=np.zeros(1),
        counts=np.zeros(1, dtype=np.int64),
        probabilities=np.zeros(1),
    )
    dataset = Dataset(matrices, [solution], copy=False)

    matrices[0, 0, 0] = 42.0
    solution.costs[0] = 1.0

    check.equal(dataset.matrices[0, 0, 0].item(), 42.0)
    check.is_(dataset.solutions[0], solution)
    check.equal(dataset.solutions[0].costs[0].item(), 1.0)


@pytest.mark.parametrize("negative_offdiag_rate", [0.0, 0.2])
def test_dataset_generation(negative_offdiag_rate: float, tmp_path: Path) -> None:
    """Test dataset is generated correctly in terms of element properties asked."""
    size = 5
    num_instances = 10
    density = 0.6
    coefficient_bounds = (-100.0, 100.0)
    seed = 42

    dataset = Dataset.from_random(
        n_matrices=num_instances,
        matrix_dim=size,
        densities=[density],
        coefficient_bounds=coefficient_bounds,
        rng=numpy_rng(seed),
        negative_offdiag_rate=negative_offdiag_rate,
    )
    assert len(dataset) == num_instances

    # test also save and load
    file_path = tmp_path / "qubo_dataset_test.bin"
    dataset.save(file_path)
    assert file_path.exists()
    loaded_data = Dataset.load(file_path)
    assert len(loaded_data) == num_instances

    off_diag = ~np.eye(size, dtype=bool)

    for qubo, _ in dataset:
        assert qubo.matrix.shape[0] == size
        assert np.isclose(_calculate_density(qubo.matrix), density, atol=1e-1)
        assert np.all(qubo.matrix >= coefficient_bounds[0])
        assert np.all(qubo.matrix <= coefficient_bounds[1])
        if negative_offdiag_rate:
            negative_off_diag = qubo.matrix[off_diag] < 0
            assert negative_off_diag.sum().item() == int(negative_offdiag_rate * size) * 2
        else:
            assert np.all(qubo.matrix[off_diag] >= 0)


def test_save_load_to_a_path_preserves_nested_solutions(tmp_path: Path) -> None:
    # Solutions are written into the dataset's own already-open stream. Saving
    # to a *path* previously reopened that path once per solution, truncating
    # everything written before it.
    dataset = Dataset(
        matrices=np.stack([np.eye(2, dtype=np.float64)] * 2, axis=-1),
        solutions=[
            Solution(
                bitstrings=np.asarray([[1, 0]], dtype=np.int8),
                costs=np.asarray([1.0]),
                counts=np.asarray([2], dtype=np.int64),
                probabilities=np.asarray([1.0]),
            ),
            Solution(
                bitstrings=np.asarray([[0, 1]], dtype=np.int8),
                costs=np.asarray([2.0]),
                counts=np.asarray([5], dtype=np.int64),
                probabilities=np.asarray([1.0]),
            ),
        ],
    )

    file_path = tmp_path / "dataset.bin"
    dataset.save(file_path)
    loaded = Dataset.load(file_path)

    check.equal(len(loaded), 2)
    check.equal(len(loaded.solutions), 2)
    for original, roundtripped in zip(dataset.solutions, loaded.solutions, strict=True):
        check.is_true(np.array_equal(roundtripped.bitstrings, original.bitstrings))
        check.is_true(np.allclose(roundtripped.costs, original.costs))
        check.is_true(np.array_equal(roundtripped.counts, original.counts))


def test_load_rejects_a_stream_that_is_not_a_qubosolver_file() -> None:
    buffer = io.BytesIO(b"not a qubosolver file at all")

    with pytest.raises(ValueError, match="Not a qubosolver file"):
        Dataset.load(buffer)


def test_getitem_returns_the_matching_solution() -> None:
    solution = Solution(bitstrings=np.zeros((1, 3), dtype=np.int8), costs=np.zeros(1))
    dataset = Dataset(np.zeros((3, 3, 1)), [solution])
    _, item_solution = dataset[0]
    np.testing.assert_allclose(item_solution.bitstrings, solution.bitstrings, rtol=0, strict=True)


# A single non-zero entry out of 4 * 4: the mask holds one diagonal coefficient only.
_DIAGONAL_ONLY_DENSITY = 1 / 16


def test_from_random_forces_a_negative_off_diagonal_coefficient() -> None:
    dataset = Dataset.from_random(
        1,
        4,
        densities=[_DIAGONAL_ONLY_DENSITY],
        negative_offdiag_rate=0.5,
        rng=numpy_rng(445670878),
    )
    instance, _ = dataset[0]
    check.less(instance.matrix[0, 1].item(), 0.0)


def test_from_random_with_positive_lower_bound_has_a_negative_diagonal() -> None:
    dataset = Dataset.from_random(1, 4, coefficient_bounds=(1.0, 2.0), rng=numpy_rng(445670878))
    instance, _ = dataset[0]
    check.is_true(bool((np.diag(instance.matrix) < 0).any()))


def test_from_random_keeps_an_existing_max_coefficient() -> None:
    # Every non-zero diagonal coefficient already equals the upper bound.
    dataset = Dataset.from_random(1, 4, coefficient_bounds=(-3.0, -3.0), rng=numpy_rng(445670878))
    instance, _ = dataset[0]
    off_diag = ~np.eye(4, dtype=bool)
    check.is_true(bool((np.diag(instance.matrix) == -3.0).any()))
    check.is_true(bool((instance.matrix[off_diag] >= 0).all()))


def test_from_random_without_positive_coefficient_sets_the_max_on_the_diagonal() -> None:
    dataset = Dataset.from_random(
        1,
        4,
        densities=[_DIAGONAL_ONLY_DENSITY],
        coefficient_bounds=(-2.0, -1.0),
        rng=numpy_rng(445670878),
    )
    instance, _ = dataset[0]
    check.is_true(bool((np.diag(instance.matrix) == -1.0).any()))


@pytest.mark.parametrize("density", [0.0, 1.0])
def test_from_random_extreme_densities(density: float) -> None:
    dataset = Dataset.from_random(1, 3, densities=[density], rng=numpy_rng(445670878))
    instance, _ = dataset[0]
    check.equal(instance.size, 3)


def test_from_random_keeps_each_density_in_its_own_slots() -> None:
    # Matrices are laid out density by density: n_matrices slots per density, in order.
    dataset = Dataset.from_random(2, 6, densities=(0.2, 1.0), rng=numpy_rng(3449876512))
    densities = [_calculate_density(instance.matrix) for instance, _ in dataset]
    check.equal(len(densities), 4)
    for actual, expected in zip(densities, (0.2, 0.2, 1.0, 1.0), strict=True):
        check.almost_equal(actual, expected, abs=0.1)
