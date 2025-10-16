from __future__ import annotations

import pytest
from qoolqit._solvers.types import BackendType, DeviceType

from qubosolver.config import (
    ClassicalConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
)
from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    DriveType,
)


def test_empty_config(empty_config: SolverConfig) -> None:
    assert empty_config.config_name == ""
    assert empty_config.use_quantum is False
    assert empty_config.backend_config.backend == BackendType.QUTIP

    assert empty_config.backend_config.device is None
    assert empty_config.backend_config.project_id is None
    assert empty_config.backend_config.username is None
    assert empty_config.backend_config.password is None
    assert empty_config.embedding.embedding_method == EmbedderType.GREEDY
    assert empty_config.embedding.draw_steps is False
    assert empty_config.embedding.greedy_layout == LayoutType.TRIANGULAR


def test_classical_part() -> None:
    default_classical = ClassicalConfig()
    assert default_classical.classical_solver_type == "cplex"
    assert default_classical.cplex_log_path == "solver.log"
    assert default_classical.cplex_maxtime == 600.0

    with pytest.raises(ValueError):
        ClassicalConfig(classical_solver_type=1)


def test_pulseshape_part() -> None:
    default_pshaper = DriveShapingConfig()
    assert default_pshaper.drive_shaping_method == DriveType.ADIABATIC
    assert not default_pshaper.re_execute_opt_pulse

    assert len(default_pshaper.optimized_initial_detuning_parameters) == 3
    assert len(default_pshaper.optimized_initial_omega_parameters) == 3

    with pytest.raises(ValueError):
        DriveShapingConfig(drive_shaping_method="dummy")


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
    assert qutip_solver_config.backend_config.backend == BackendType.QUTIP


def test_blade_config(blade_config: SolverConfig) -> None:
    assert blade_config.embedding.embedding_method == EmbedderType.BLADE
    assert (
        blade_config.backend_config.device == DeviceType.DIGITAL_ANALOG_DEVICE
        and blade_config.embedding.blade_dimensions == [2]
    )
    assert blade_config.embedding.blade_dimensions == [2]


def test_blade_clear_dimensions_config(
    blade_clear_dimensions_config: SolverConfig,
) -> None:
    assert blade_clear_dimensions_config.embedding.blade_dimensions == [6, 5, 4, 3, 2]


def test_greedy_embedding_config(greedy_embedding_config: SolverConfig) -> None:
    assert greedy_embedding_config.embedding.embedding_method == EmbedderType.GREEDY
    assert greedy_embedding_config.backend_config.device == DeviceType.DIGITAL_ANALOG_DEVICE
    assert greedy_embedding_config.embedding.greedy_layout == LayoutType.SQUARE
    assert greedy_embedding_config.embedding.greedy_traps == 10
    assert greedy_embedding_config.embedding.greedy_spacing == 5.0


def test_initialization_device() -> None:
    from qoolqit._solvers.types import DeviceType

    solver = SolverConfig()
    assert solver.embedding.greedy_traps == DeviceType.DIGITAL_ANALOG_DEVICE.value.min_layout_traps
    assert solver.embedding.greedy_spacing == float(
        DeviceType.DIGITAL_ANALOG_DEVICE.value.min_atom_distance
    )

    solver = SolverConfig.from_kwargs(**{"device": DeviceType.ANALOG_DEVICE})
    assert solver.embedding.greedy_traps == DeviceType.ANALOG_DEVICE.value.min_layout_traps
    assert solver.embedding.greedy_spacing == float(
        DeviceType.ANALOG_DEVICE.value.min_atom_distance
    )
