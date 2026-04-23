from __future__ import annotations

import pytest
from typing import Any
import pytest_check as check
from pulser.devices import DigitalAnalogDevice as PulserDADevice
from pulser_simulation import QutipBackendV2
from qoolqit.devices.device import AnalogDevice
from qubosolver.config import (
    LocalEmulator,
    ClassicalConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    DecompositionConfig,
    SolverConfig,
)
from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    DriveType,
    ClassicalSolverType,
)


def test_empty_config(empty_config: SolverConfig) -> None:
    assert empty_config.config_name == ""
    assert empty_config.use_quantum is True
    assert isinstance(empty_config.backend, LocalEmulator)
    assert empty_config.backend._backend_type == QutipBackendV2
    assert empty_config.embedding.embedding_method == EmbedderType.GREEDY
    assert empty_config.embedding.draw_steps is False
    assert empty_config.embedding.greedy_layout == LayoutType.TRIANGULAR
    assert empty_config.decompose is None


def test_classical_part() -> None:
    default_classical = ClassicalConfig()
    assert (
        default_classical.classical_solver_type
        == ClassicalSolverType.SIMULATED_ANNEALING_TABU_SEARCH
    )

    with pytest.raises(ValueError):
        ClassicalConfig(classical_solver_type=1)  # type: ignore[arg-type]


def test_pulseshape_part() -> None:
    default_pshaper = DriveShapingConfig()
    assert default_pshaper.drive_shaping_method == DriveType.HEURISTIC
    assert not default_pshaper.optimized_re_execute_opt_drive

    assert len(default_pshaper.optimized_initial_detuning_parameters) == 3
    assert len(default_pshaper.optimized_initial_omega_parameters) == 3

    with pytest.raises(ValueError):
        DriveShapingConfig(drive_shaping_method="dummy")

    check.equal(
        DriveShapingConfig(drive_shaping_method="heuristic").drive_shaping_method,
        DriveType.HEURISTIC,
    )
    check.equal(
        DriveShapingConfig(drive_shaping_method="optimized").drive_shaping_method,
        DriveType.OPTIMIZED,
    )


def test_embedder_part() -> None:
    default_embedder = EmbeddingConfig()
    assert default_embedder.embedding_method == EmbedderType.GREEDY
    assert default_embedder.draw_steps is False
    assert default_embedder.greedy_layout == LayoutType.TRIANGULAR
    assert default_embedder.greedy_traps

    with pytest.raises(ValueError):
        EmbeddingConfig(embedding_method="dummy")
    with pytest.raises(ValueError):
        EmbeddingConfig(greedy_layout="dummy")


def test_config_name(name_config: SolverConfig) -> None:
    assert name_config.config_name == "my_config"


def test_classical_config_flag(classical_solver_config: SolverConfig) -> None:
    assert classical_solver_config.use_quantum is False


def test_qutip_config_backend(qutip_solver_config: SolverConfig) -> None:
    assert qutip_solver_config.backend._backend_type == QutipBackendV2


def test_blade_config(blade_config: SolverConfig) -> None:
    assert blade_config.embedding.embedding_method == EmbedderType.BLADE
    assert (
        blade_config.device._device == PulserDADevice
        and blade_config.embedding.blade_dimensions == [2]
    )
    assert blade_config.embedding.blade_dimensions == [2]


def test_blade_clear_dimensions_config(
    blade_clear_dimensions_config: SolverConfig,
) -> None:
    assert blade_clear_dimensions_config.embedding.blade_dimensions == [6, 5, 4, 3, 2]


def test_greedy_embedding_config(greedy_embedding_config: SolverConfig) -> None:
    assert greedy_embedding_config.embedding.embedding_method == EmbedderType.GREEDY
    assert greedy_embedding_config.device._device == PulserDADevice
    assert greedy_embedding_config.embedding.greedy_layout == LayoutType.SQUARE
    assert greedy_embedding_config.embedding.greedy_traps == 10
    assert greedy_embedding_config.embedding.greedy_spacing == 5.0


def test_initialization_device() -> None:

    solver = SolverConfig()
    assert solver.embedding.greedy_traps == -1
    assert solver.embedding.greedy_spacing == 7.0

    deviceanalog = AnalogDevice()
    kwargs: dict[Any, Any] = {"device": deviceanalog}
    solver = SolverConfig.from_kwargs(**kwargs)
    assert solver.embedding.greedy_traps == -1
    assert solver.embedding.greedy_spacing == 7.0


def test_decomposition_config() -> None:
    config = SolverConfig(decompose=DecompositionConfig())
    assert config.decompose is not None

    config_kwargs = SolverConfig.from_kwargs(**config.decompose.model_dump())
    assert config_kwargs.decompose is not None
