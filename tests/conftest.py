# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

import pytest
import torch

from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend

from qoolqit.devices import DigitalAnalogDevice, AnalogDevice, MockDevice, Device
from pulser_pasqal import PasqalCloud
from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.config import (
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
    LocalEmulator,
)
from qubosolver.qubo_types import EmbedderType, LayoutType, DriveType

connection = PasqalCloud()
connection.fetch_available_devices()


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
def basic_solution() -> QUBOSolution:
    return QUBOSolution(
        bitstrings=torch.tensor([[0, 1, 0], [1, 0, 1]]),
        costs=torch.tensor([1.0, 2.0]),
        counts=torch.tensor([15, 5]),
        probabilities=torch.tensor([0.75, 0.25]),
    )


@pytest.fixture
def analyzer(basic_solution: QUBOSolution) -> QUBOAnalyzer:
    return QUBOAnalyzer(solutions=[basic_solution], labels=["sol1"])


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
    LocalEmulator(backend_type=btype, runs=500)
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
        DigitalAnalogDevice(),
        AnalogDevice(),
        MockDevice(),
        Device(pulser_device=connection.fetch_available_devices()["FRESNEL"]),
    ],
    ids=[
        "DigitalAnalogDevice",
        "AnalogDevice",
        "MockDevice",
        "FRESNEL",
    ],
)
def local_device(request: pytest.FixtureRequest) -> Device:
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
        backend=LocalEmulator(backend_type=QutipBackendV2, runs=500),
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
        greedy_spacing=5.0,
    )
    return SolverConfig(
        embedding=embed_method,
    )


@pytest.fixture
def qubo_instance_for_preprocessing() -> QUBOInstance:
    """
    Generate small instance for pre/postprocessing.
    """
    return QUBOInstance(
        torch.tensor(
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
def simple_qubo_instance() -> QUBOInstance:
    Q = torch.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])
    return QUBOInstance(coefficients=Q)


@pytest.fixture
def simple_qubo_instance2() -> QUBOInstance:
    Q = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    return QUBOInstance(coefficients=Q)


@pytest.fixture
def qubo_instance_for_embedding() -> QUBOInstance:
    """
    Small QUBO instance for embedding.
    """
    return QUBOInstance(
        torch.tensor(
            [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
            dtype=torch.int32,
        )
    )


@pytest.fixture
def qubo_instance_blade_tutorial() -> QUBOInstance:
    return QUBOInstance(
        torch.tensor(
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
    )


@pytest.fixture
def qubo_instance_adiabatic_tutorial() -> QUBOInstance:
    return QUBOInstance(
        torch.tensor(
            [
                [
                    -63.9423,
                    0.0000,
                    73.6471,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    55.2853,
                ],
                [0.0000, -44.1916, 0.0000, 0.0000, 0.0000, 0.0000, 58.9307, 0.0000, 0.0000, 0.0000],
                [
                    73.6471,
                    0.0000,
                    -89.8861,
                    51.0382,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                ],
                [
                    0.0000,
                    0.0000,
                    51.0382,
                    -63.7618,
                    0.0000,
                    0.0000,
                    33.9093,
                    0.0000,
                    0.0000,
                    0.0000,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    -94.4426,
                    18.7963,
                    0.0000,
                    0.0000,
                    14.3994,
                    0.0000,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    18.7963,
                    -60.7545,
                    0.0000,
                    0.0000,
                    0.0000,
                    96.9903,
                ],
                [
                    0.0000,
                    58.9307,
                    0.0000,
                    33.9093,
                    0.0000,
                    0.0000,
                    -71.3241,
                    0.0000,
                    0.0000,
                    0.0000,
                ],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -38.2094, 59.3175, 0.0000],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    14.3994,
                    0.0000,
                    0.0000,
                    59.3175,
                    -94.5790,
                    18.0653,
                ],
                [
                    55.2853,
                    0.0000,
                    0.0000,
                    0.0000,
                    0.0000,
                    96.9903,
                    0.0000,
                    0.0000,
                    18.0653,
                    -97.3174,
                ],
            ]
        )
    )


@pytest.fixture(
    params=[
        "simple_qubo_instance",
        "simple_qubo_instance2",
        "qubo_instance_for_embedding",
        "qubo_instance_adiabatic_tutorial",
        "qubo_instance_blade_tutorial",
    ],
)
def qubo_for_testing_many_devices(request: pytest.FixtureRequest) -> QUBOInstance:
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def generate_qubo_matrix(
    size: int, density: float, value_range: tuple[int, int], seed: int | None = None
) -> torch.Tensor:
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
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, i] = -np.abs(np.random.uniform(0, 100))  # Negative diagonal
        for j in range(i + 1, size):
            if np.random.rand() < density:
                value = np.abs(
                    np.random.uniform(value_range[0], value_range[1])
                )  # Positive off-diagonal
                matrix[i, j] = value
                matrix[j, i] = value
    return torch.tensor(matrix)


@pytest.fixture
def decomposable_qubo() -> QUBOInstance:
    return QUBOInstance(generate_qubo_matrix(50, 0.30, (0, 20), 1))
