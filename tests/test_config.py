from __future__ import annotations

import pytest
import pytest_check as check
from pulser_simulation import QutipBackendV2
from qoolqit import AnalogDeviceWithDMM
from qubosolver import (
    SolverConfig,
    QuantumSolvingConfig,
    ClassicalSolvingConfig,
    DriveShapingConfig,
    EmbeddingConfig,
    DecompositionConfig,
    LocalEmulator,
    AutoLocalEmulatorBackend,
)


def test_default_config() -> None:
    default_config = SolverConfig()
    check.equal(default_config.config_name, "")
    check.equal(default_config.solving_mode, "quantum")
    assert isinstance(default_config.solving, QuantumSolvingConfig)
    check.is_instance(default_config.solving.backend, LocalEmulator)
    check.is_(default_config.solving.backend._backend_type, AutoLocalEmulatorBackend)
    check.equal(default_config.solving.embedding.algorithm, "greedy_layout")
    check.equal(default_config.solving.embedding.greedy_layout_lattice, "triangular")
    check.is_none(default_config.decompose)


def test_default_classical_config() -> None:
    default_classical = ClassicalSolvingConfig()
    check.equal(default_classical.algorithm, "tabu_search")
    with pytest.raises(ValueError):
        ClassicalSolvingConfig(algorithm=1)  # type: ignore[arg-type]


def test_drive_shaping_config() -> None:
    default_drive_shaping_config = DriveShapingConfig()
    check.equal(default_drive_shaping_config.algorithm, "proportional_diagonal")

    check.equal(len(default_drive_shaping_config.bayesian_search_initial_detuning_parameters), 3)
    check.equal(len(default_drive_shaping_config.bayesian_search_initial_omega_parameters), 3)

    with pytest.raises(ValueError):
        DriveShapingConfig(algorithm="dummy")  # type: ignore[arg-type]

    check.equal(
        DriveShapingConfig(algorithm="proportional_diagonal").algorithm,
        "proportional_diagonal",
    )
    check.equal(
        DriveShapingConfig(algorithm="bayesian_search").algorithm,
        "bayesian_search",
    )


def test_embdedding_config() -> None:
    default_embedding_config = EmbeddingConfig()
    check.equal(default_embedding_config.algorithm, "greedy_layout")
    check.equal(default_embedding_config.greedy_layout_lattice, "triangular")
    check.is_true(default_embedding_config.greedy_layout_traps)

    with pytest.raises(ValueError):
        EmbeddingConfig(algorithm="dummy")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EmbeddingConfig(greedy_layout_lattice="dummy")  # type: ignore[arg-type]


def test_config_name() -> None:
    name_config = SolverConfig(config_name="my_config")
    check.equal(name_config.config_name, "my_config")


def test_classical_config_flag() -> None:
    classical_solver_config = SolverConfig(solving=ClassicalSolvingConfig())
    check.equal(classical_solver_config.solving_mode, "classical")


def test_solving_mode() -> None:
    quantum_solver_config = SolverConfig(solving=QuantumSolvingConfig())
    check.equal(quantum_solver_config.solving_mode, "quantum")

    classical_solver_config = SolverConfig(solving=ClassicalSolvingConfig())
    check.equal(classical_solver_config.solving_mode, "classical")


def test_solving_mode_non_default_config() -> None:
    classical_solver_config = SolverConfig(
        solving=ClassicalSolvingConfig(algorithm="simulated_annealing", max_iter=500),
    )
    check.equal(classical_solver_config.solving_mode, "classical")

    quantum_solver_config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm="blade", blade_dimensions=[2]),
        ),
    )
    check.equal(quantum_solver_config.solving_mode, "quantum")


def test_quantum_config_property() -> None:
    quantum_solver_config = SolverConfig(solving=QuantumSolvingConfig())
    check.is_(quantum_solver_config.quantum, quantum_solver_config.solving)

    classical_solver_config = SolverConfig(solving=ClassicalSolvingConfig())
    with pytest.raises(ValueError):
        classical_solver_config.quantum


def test_classical_config_property() -> None:
    classical_solver_config = SolverConfig(solving=ClassicalSolvingConfig())
    check.is_(classical_solver_config.classical, classical_solver_config.solving)

    quantum_solver_config = SolverConfig(solving=QuantumSolvingConfig())
    with pytest.raises(ValueError):
        quantum_solver_config.classical


def test_qutip_config_backend() -> None:
    qutip_solver_config = SolverConfig(
        solving=QuantumSolvingConfig(
            backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
        ),
    )
    assert isinstance(qutip_solver_config.solving, QuantumSolvingConfig)
    check.is_(qutip_solver_config.solving.backend._backend_type, QutipBackendV2)


def test_blade_config() -> None:
    embed_method = EmbeddingConfig(algorithm="blade", blade_dimensions=[2])
    blade_config = QuantumSolvingConfig(embedding=embed_method)
    check.equal(blade_config.embedding.algorithm, "blade")
    check.equal(type(blade_config.device), AnalogDeviceWithDMM)
    check.equal(blade_config.embedding.blade_dimensions, [2])


def test_blade_clear_dimensions_config() -> None:
    embed_method = EmbeddingConfig(blade_dimensions=[6, 5, 4, 3, 2])
    blade_clear_dimensions_config = QuantumSolvingConfig(embedding=embed_method)
    check.equal(blade_clear_dimensions_config.embedding.blade_dimensions, [6, 5, 4, 3, 2])


def test_greedy_embedding_config() -> None:

    embedding_config = EmbeddingConfig(
        algorithm="greedy_layout",
        greedy_layout_lattice="square",
        greedy_layout_traps=10,
    )
    config = QuantumSolvingConfig(
        embedding=embedding_config,
    )
    check.equal(config.embedding.algorithm, "greedy_layout")
    check.is_instance(config.device, AnalogDeviceWithDMM)
    check.equal(config.embedding.greedy_layout_lattice, "square")
    check.equal(config.embedding.greedy_layout_traps, 10)


def test_initialization_device() -> None:

    solver = QuantumSolvingConfig()
    check.equal(solver.embedding.greedy_layout_traps, "device")


def test_decomposition_config() -> None:
    config = SolverConfig(decompose=DecompositionConfig())
    check.is_not_none(config.decompose)
