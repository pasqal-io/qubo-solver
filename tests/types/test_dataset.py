from __future__ import annotations

import io
import pytest
import pytest_check as check
import torch
import os
from pathlib import Path

import numpy as np
from qubosolver import Dataset, torch_rng, bitstrings, vector, vectori
from qubosolver.types.instance import _calculate_density
from qubosolver.types.solution import Solution


def test_dataset_copies_input_matrices() -> None:
    """Test that Dataset copies the input tensor instead of aliasing it."""

    matrices = torch.zeros(3, 3, 2)
    dataset = Dataset(matrices)

    matrices[0, 0, 0] = 42.0

    check.equal(dataset.matrices[0, 0, 0].item(), 0.0)


def test_dataset_copies_input_solutions() -> None:
    """Test that Dataset deep-copies the input solutions instead of aliasing them."""

    matrices = torch.zeros(3, 3, 1)
    solution = Solution(
        bitstrings=bitstrings.zeros(1, 3),
        costs=vector.zeros(1),
        counts=vectori.zeros(1),
        probabilities=vector.zeros(1),
    )
    dataset = Dataset(matrices, [solution])

    solution.bitstrings[0, 0] = 1
    solution.costs[0] = 42.0

    check.equal(dataset.solutions[0].bitstrings[0, 0].item(), 0)
    check.equal(dataset.solutions[0].costs[0].item(), 0.0)


def test_dataset_copy_false_aliases_input() -> None:
    """Test that Dataset(copy=False) stores the given matrices/solutions directly."""

    matrices = torch.zeros(3, 3, 1)
    solution = Solution(
        bitstrings=torch.zeros(1, 3, dtype=torch.int8),
        costs=torch.zeros(1),
        counts=torch.zeros(1, dtype=torch.int64),
        probabilities=torch.zeros(1),
    )
    dataset = Dataset(matrices, [solution], copy=False)

    matrices[0, 0, 0] = 42.0
    solution.costs[0] = 1.0

    check.equal(dataset.matrices[0, 0, 0].item(), 42.0)
    check.is_(dataset.solutions[0], solution)
    check.equal(dataset.solutions[0].costs[0].item(), 1.0)


@pytest.mark.parametrize("negative_offdiag_rate", [0.0, 0.2])
def test_dataset_generation(negative_offdiag_rate: float) -> None:
    """Test dataset is generated correctly in terms of element properties asked"""

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
        rng=torch_rng(seed),
        negative_offdiag_rate=negative_offdiag_rate,
    )
    assert len(dataset) == num_instances

    # test also save and load
    file_path = Path(__file__).parent / "qubo_dataset_test.pt"
    dataset.save(file_path)
    assert os.path.exists(file_path)
    loaded_data = Dataset.load(file_path)
    assert len(loaded_data) == num_instances
    if os.path.exists(file_path):
        os.remove(file_path)

    off_diag = ~torch.eye(size, dtype=torch.bool)

    for qubo, _ in dataset:
        assert qubo.matrix.shape[0] == size
        assert np.isclose(_calculate_density(qubo.matrix), density, atol=1e-1)
        assert torch.all(qubo.matrix >= coefficient_bounds[0])
        assert torch.all(qubo.matrix <= coefficient_bounds[1])
        if negative_offdiag_rate:
            negative_off_diag = qubo.matrix[off_diag] < 0
            assert torch.any(negative_off_diag) or torch.all(qubo.matrix[off_diag] == 0)
            if torch.any(negative_off_diag):
                assert negative_off_diag.sum().item() == int(negative_offdiag_rate * size) * 2
        else:
            assert torch.all(qubo.matrix[off_diag] >= 0)


def test_save_load_to_a_path_preserves_nested_solutions(tmp_path: Path) -> None:
    # Solutions are written into the dataset's own already-open stream. Saving
    # to a *path* previously reopened that path once per solution, truncating
    # everything written before it.
    dataset = Dataset(
        matrices=torch.stack([torch.eye(2), torch.eye(2)], dim=-1),
        solutions=[
            Solution(
                bitstrings=bitstrings.tensor([[1, 0]]),
                costs=vector.tensor([1.0]),
                counts=vectori.tensor([2]),
                probabilities=vector.tensor([1.0]),
            ),
            Solution(
                bitstrings=bitstrings.tensor([[0, 1]]),
                costs=vector.tensor([2.0]),
                counts=vectori.tensor([5]),
                probabilities=vector.tensor([1.0]),
            ),
        ],
    )

    file_path = tmp_path / "dataset.bin"
    dataset.save(file_path)
    loaded = Dataset.load(file_path)

    check.equal(len(loaded), 2)
    check.equal(len(loaded.solutions), 2)
    for original, roundtripped in zip(dataset.solutions, loaded.solutions):
        check.is_true(torch.equal(roundtripped.bitstrings, original.bitstrings))
        check.is_true(torch.allclose(roundtripped.costs, original.costs))
        check.is_true(torch.equal(roundtripped.counts, original.counts))


def test_load_rejects_a_stream_that_is_not_a_qubosolver_file() -> None:
    buffer = io.BytesIO(b"not a qubosolver file at all")

    with pytest.raises(ValueError, match="Not a qubosolver file"):
        Dataset.load(buffer)
