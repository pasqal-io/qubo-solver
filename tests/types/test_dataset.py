from __future__ import annotations

import pytest
import torch
import os
from pathlib import Path

import numpy as np
from qubosolver import Dataset, torch_rng
from qubosolver.types.instance import _calculate_density


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
    Dataset.save(dataset, file_path)
    assert os.path.exists(file_path)
    loaded_data = Dataset.load(file_path)
    assert len(loaded_data) == num_instances
    if os.path.exists(file_path):
        os.remove(file_path)

    off_diag = ~torch.eye(size, dtype=torch.bool)

    for qubo, _ in dataset:
        assert qubo.shape[0] == size
        assert np.isclose(_calculate_density(qubo), density, atol=1e-1)
        assert torch.all(qubo >= coefficient_bounds[0])
        assert torch.all(qubo <= coefficient_bounds[1])
        if negative_offdiag_rate:
            negative_off_diag = qubo[off_diag] < 0
            assert torch.any(negative_off_diag) or torch.all(qubo[off_diag] == 0)
            if torch.any(negative_off_diag):
                assert negative_off_diag.sum().item() == int(negative_offdiag_rate * size) * 2
        else:
            assert torch.all(qubo[off_diag] >= 0)
