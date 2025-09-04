from __future__ import annotations

import pytest
from qoolqit._solvers.types import BackendType, DeviceType

from qubosolver.config import (
    ClassicalConfig,
    EmbeddingConfig,
    PulseShapingConfig,
    SolverConfig,
)
from qubosolver.qubo_types import (
    EmbedderType,
    LayoutType,
    PulseType,
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
    assert empty_config.embedding.layout_greedy_embedder == LayoutType.TRIANGULAR


def test_classical_part() -> None:
    default_classical = ClassicalConfig()
    assert default_classical.classical_solver_type == "cplex"
    assert default_classical.cplex_log_path == "solver.log"
    assert default_classical.cplex_maxtime == 600.0

    with pytest.raises(ValueError):
        ClassicalConfig(classical_solver_type=1)


def test_pulseshape_part() -> None:
    default_pshaper = PulseShapingConfig()
    assert default_pshaper.pulse_shaping_method == PulseType.ADIABATIC
    assert not default_pshaper.re_execute_opt_pulse

    assert len(default_pshaper.initial_detuning_parameters) == 3
    assert len(default_pshaper.initial_omega_parameters) == 3

    with pytest.raises(ValueError):
        PulseShapingConfig(pulse_shaping_method="dummy")


def test_embedder_part() -> None:
    default_embedder = EmbeddingConfig()
    assert default_embedder.embedding_method == EmbedderType.GREEDY
    assert default_embedder.draw_steps is False
    assert default_embedder.layout_greedy_embedder == LayoutType.TRIANGULAR
    assert default_embedder.traps

    with pytest.raises(ValueError):
        EmbeddingConfig(embedding_method="dummy")
    with pytest.raises(ValueError):
        EmbeddingConfig(layout_greedy_embedder="dummy")


def test_config_name(name_config: SolverConfig) -> None:
    assert name_config.config_name == "my_config"


def test_classical_config_flag(classical_solver_config: SolverConfig) -> None:
    assert classical_solver_config.use_quantum is False


def test_qutip_config_backend(qutip_solver_config: SolverConfig) -> None:
    assert qutip_solver_config.backend_config.backend == BackendType.QUTIP


def test_greedy_embedding_config(greedy_embedding_config: SolverConfig) -> None:
    assert greedy_embedding_config.embedding.embedding_method == EmbedderType.GREEDY
    assert (
        greedy_embedding_config.backend_config.device
        == DeviceType.DIGITAL_ANALOG_DEVICE
    )
    assert greedy_embedding_config.embedding.layout_greedy_embedder == LayoutType.SQUARE
    assert greedy_embedding_config.embedding.traps == 10
    assert greedy_embedding_config.embedding.spacing == 5.0


def test_initialization_device() -> None:
    from qoolqit._solvers.types import DeviceType

    solver = SolverConfig()
    assert (
        solver.embedding.traps
        == DeviceType.DIGITAL_ANALOG_DEVICE.value.min_layout_traps
    )
    assert solver.embedding.spacing == float(
        DeviceType.DIGITAL_ANALOG_DEVICE.value.min_atom_distance
    )

    solver = SolverConfig.from_kwargs(**{"device": DeviceType.ANALOG_DEVICE})
    assert solver.embedding.traps == DeviceType.ANALOG_DEVICE.value.min_layout_traps
    assert solver.embedding.spacing == float(
        DeviceType.ANALOG_DEVICE.value.min_atom_distance
    )
