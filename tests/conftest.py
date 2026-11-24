# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

# use this file for configuring test fixtures and
# functions common to every test
# ruff: noqa: E402 (thread limits must be set before numpy/torch are imported)
from __future__ import annotations

import os

# One BLAS/OpenMP thread per xdist worker: otherwise each worker starts one thread per core
# and the CPU is heavily oversubscribed. Must be set before numpy/torch are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import qoolqit
import torch
from mock.connection import MockConnection

from qubosolver import (
    Instance,
    Matrix,
    Solution,
    bitstrings,
    matrix,
    vector,
    vectori,
)

if TYPE_CHECKING:
    from collections.abc import Generator

SLOW_THRESHOLD_S = 5

# Not unit tests: they must not contribute to coverage (see `omit` in pyproject.toml).
_TESTS_DIR = Path(__file__).parent
NO_COVER_PATHS = tuple(_TESTS_DIR / p for p in ("pipeline", "integration", "test_notebooks.py"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Order collected tests longest-first and tag them `fast` or `slow`.

    This hook is called by pytest after test collection and before execution.

    Marker convention:
        @pytest.mark.duration(<seconds>)

    Behavior:
    - Tests are sorted in-place by their expected runtime, longest first, so
      long-running tests start earlier and overall wall-clock time is reduced.
    - Tests without a `duration` marker are treated as duration 0.
    - Tests with duration >= `SLOW_THRESHOLD_S` get the `slow` marker, the
      others get `fast`, so `pytest -m fast` / `pytest -m slow` select them.
      Do not apply `fast` / `slow` manually; set `duration` instead.
    - Tests under `NO_COVER_PATHS` get pytest-cov's `no_cover` marker, so they
      do not contribute to coverage.
    - Runs `tryfirst` so the markers exist before pytest's `-m` deselection.

    Args:
        items (list[pytest.Item]):
            The list of collected test items to be executed. This list is
            mutated in-place.
    """

    def duration(item: pytest.Item) -> int:
        marker = item.get_closest_marker("duration")
        return int(marker.args[0]) if marker else 0

    items.sort(key=duration, reverse=True)
    for item in items:
        item.add_marker(
            pytest.mark.slow if duration(item) >= SLOW_THRESHOLD_S else pytest.mark.fast
        )
        if any(item.path.is_relative_to(p) for p in NO_COVER_PATHS):
            item.add_marker(pytest.mark.no_cover)

    # QUBOSOLVER_WITHOUT_EXTRAS explicitly declares that extras (e.g. cplex) are
    # not installed, so extras-marked tests are skipped rather than failing.
    # When unset, extras are assumed to be installed; if they're not, the
    # extras-marked tests run and fail naturally with the install instructions
    # already raised by the code under test (see cplex.py's _import_cplex()).
    if os.environ.get("QUBOSOLVER_WITHOUT_EXTRAS"):  # no cov
        skip_extras = pytest.mark.skip(
            reason="QUBOSOLVER_WITHOUT_EXTRAS is set: extras (e.g. cplex) are not installed"
        )
        for item in items:
            if "extras" in item.keywords:
                item.add_marker(skip_extras)


@pytest.fixture
def basic_solution() -> Solution:
    return Solution(
        bitstrings=bitstrings.tensor([[0, 1, 0], [1, 0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([15, 5]),
        probabilities=vector.tensor([0.75, 0.25]),
    )


@pytest.fixture
def simple_qubo_instance() -> Instance:
    Q = matrix.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])
    return Instance(matrix=Q)


def generate_qubo_matrix(
    size: int, density: float, value_range: tuple[int, int], seed: int
) -> Matrix:
    """Generate a random symmetric qubo matrix with negative diagonal coefficients.

    Off-diagonal elements are positive.

    Args:
        size (int): Size of qubo.
        density (float): Density.
        value_range (tuple[int, int]): Value range of elements.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Qubo matrix.
    """
    np.random.seed(seed)
    matrix_ = np.zeros((size, size))
    for i in range(size):
        matrix_[i, i] = -np.abs(np.random.uniform(0, 100))  # Negative diagonal
        for j in range(i + 1, size):
            if np.random.rand() < density:
                value = np.abs(
                    np.random.uniform(value_range[0], value_range[1])
                )  # Positive off-diagonal
                matrix_[i, j] = value
                matrix_[j, i] = value
    return matrix.tensor(matrix_)


@pytest.fixture
def decomposable_qubo() -> Instance:
    return Instance(generate_qubo_matrix(50, 0.30, (0, 20), 1))


@pytest.fixture
def small_decomposable_qubo() -> Instance:
    # Hansen fixing leaves all 12 variables of this one.
    return Instance(generate_qubo_matrix(12, 0.50, (0, 20), 2))


@pytest.fixture
def restore_rng_state() -> Generator:
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    yield  # run the test

    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)


@pytest.fixture
def make_mock_connection() -> type[MockConnection]:
    return MockConnection


@pytest.fixture
def device() -> qoolqit.devices.device.BaseDevice:
    return qoolqit.AnalogDeviceWithDMM()._device


@pytest.fixture
def max_min_dist_ratio(device: qoolqit.devices.device.BaseDevice) -> float:
    assert isinstance(device.max_radial_distance, int)
    assert isinstance(device.min_atom_distance, int)
    return device.max_radial_distance / device.min_atom_distance
