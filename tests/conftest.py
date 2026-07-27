# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

import pytest
import random
import torch
import numpy as np
from typing import Generator

from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
import qoolqit

from mock.connection import MockConnection

from qubosolver import (
    Instance,
    Solution,
    Analyzer,
    EmbedderType,
    LayoutType,
    DriveType,
    bitstrings,
    vector,
    vectori,
    matrix,
    Matrix,
    SolverConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    LocalEmulator,
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """
        Reorder collected pytest items so higher-priority tests run first.

        This hook is called by pytest after test collection and before execution.
        It sorts the collected `items` list in-place by the numeric value of the
        closest `priority` marker attached to each test.

        Marker convention:
                @pytest.mark.priority(<int>)

        Behavior:
        - Tests with a larger priority value are executed earlier (descending order).
        - Tests without a `priority` marker are treated as priority 0.

    Example (prioritizing long tests using estimated duration in seconds):
                Use the test's expected runtime (in seconds) as the `priority` value so
                long-running tests start earlier and overall wall-clock time is reduced.

    Args:
        items (list[pytest.Item]):
            The list of collected test items to be executed. This list is
            mutated in-place.
    """

    def priority(item: pytest.Item) -> int:
        marker = item.get_closest_marker("priority")
        return int(marker.args[0]) if marker else 0

    items.sort(key=priority, reverse=True)


@pytest.fixture
def basic_solution() -> Solution:
    return Solution(
        bitstrings=bitstrings.tensor([[0, 1, 0], [1, 0, 1]]),
        costs=vector.tensor([1.0, 2.0]),
        counts=vectori.tensor([15, 5]),
        probabilities=vector.tensor([0.75, 0.25]),
    )


@pytest.fixture
def analyzer(basic_solution: Solution) -> Analyzer:
    return Analyzer(solutions=[basic_solution], labels=["sol1"])


@pytest.fixture
def empty_config() -> SolverConfig:
    return SolverConfig()


@pytest.fixture
def name_config() -> SolverConfig:
    return SolverConfig(config_name="my_config")


@pytest.fixture
def classical_solver_config() -> SolverConfig:
    return SolverConfig(use_quantum=False)


locals_bkds: list[LocalEmulator] = [
    LocalEmulator(backend_type=btype, num_shots=500)
    for btype in [
        QutipBackendV2,
        SVBackend,
        MPSBackend,
    ]
]


@pytest.fixture(
    params=locals_bkds,
)
def local_backend(request: pytest.FixtureRequest) -> LocalEmulator:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        qoolqit.AnalogDeviceWithDMM(),
        qoolqit.AnalogDevice(),
        qoolqit.MockDevice(),
    ],
    ids=[
        "AnalogDeviceWithDMM",
        "AnalogDevice",
        "MockDevice",
    ],
)
def local_device(request: pytest.FixtureRequest) -> qoolqit.Device:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        EmbedderType.GREEDY,
        EmbedderType.BLADE,
    ]
)
def embedding_method(request: pytest.FixtureRequest) -> EmbedderType:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def qutip_solver_config() -> SolverConfig:
    return SolverConfig(
        use_quantum=True,
        backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
    )


@pytest.fixture
def blade_config() -> SolverConfig:
    embed_method = EmbeddingConfig(embedding_method="blade", blade_dimensions=[2])
    return SolverConfig(
        embedding=embed_method,
    )


@pytest.fixture
def optimized_drive_shaping() -> DriveShapingConfig:
    return DriveShapingConfig(drive_shaping_method=DriveType.OPTIMIZED)


@pytest.fixture
def blade_clear_dimensions_config() -> SolverConfig:
    embed_method = EmbeddingConfig(blade_dimensions=[6, 5, 4, 3, 2])
    return SolverConfig(embedding=embed_method)


@pytest.fixture
def greedy_embedding_config() -> SolverConfig:
    embed_method = EmbeddingConfig(
        embedding_method="greedy",
        greedy_layout=LayoutType.SQUARE,
        greedy_traps=10,
    )
    return SolverConfig(
        embedding=embed_method,
    )


@pytest.fixture
def qubo_instance_for_preprocessing() -> Instance:
    """
    Generate small instance for pre/postprocessing.
    """
    return Instance(
        matrix.tensor(
            [
                [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
                [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
                [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
                [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
                [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
            ]
        )
    )


@pytest.fixture
def simple_qubo_instance() -> Instance:
    Q = matrix.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])
    return Instance(matrix=Q)


@pytest.fixture
def simple_qubo_instance2() -> Instance:
    Q = matrix.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    return Instance(matrix=Q)


@pytest.fixture
def qubo_instance_for_embedding() -> Instance:
    """
    Small QUBO instance for embedding.
    """
    return Instance(
        matrix.tensor(
            [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
        )
    )


@pytest.fixture
def qubo_instance_blade_tutorial() -> Instance:
    M = matrix.tensor(
        [
            [0.0, 3.0, 13.0, 211.0, 49.0, 5.0, 12.0, 0.0, 0.0],
            [0.0, 0.0, 23.0, 0.0, 0.0, 4.0, 0.0, 63.0, 2.0],
            [0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 37.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 34.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 9.0, 34.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    Q = M + M.T
    return Instance(matrix=Q)


@pytest.fixture(
    params=[
        "simple_qubo_instance",
        "simple_qubo_instance2",
        "qubo_instance_for_embedding",
        "qubo_instance_blade_tutorial",
    ],
)
def qubo_for_testing_many_devices(request: pytest.FixtureRequest) -> Instance:
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def generate_qubo_matrix(
    size: int, density: float, value_range: tuple[int, int], seed: int | None = None
) -> Matrix:
    """Generate a random symmetric qubo matrix with negative diagonal coefficients
       and positive off-diagonal elements.

    Args:
        size (int): Size of qubo.
        density (float): Density.
        value_range (tuple[int, int]): Value range of elements.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        torch.Tensor: Qubo matrix.
    """

    import numpy as np

    if seed is not None:
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
