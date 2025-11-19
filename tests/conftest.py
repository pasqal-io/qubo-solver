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
from qubosolver.qubo_types import LayoutType, DriveType

connection = PasqalCloud()
connection.fetch_available_devices()


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
def local_backend(request: pytest.Fixture) -> LocalEmulator:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        DigitalAnalogDevice(),
        AnalogDevice(),
        MockDevice(),
        Device(pulser_device=connection.fetch_available_devices()["FRESNEL"]),
    ],
)
def local_device(request: pytest.Fixture) -> Device:
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
