from __future__ import annotations

import pytest
from typing import Any
import pytest_check as check
from pulser_simulation import QutipBackendV2
from qoolqit import AnalogDevice, AnalogDeviceWithDMM
from qubosolver import (
    solving,
    drive_shaping,
    embedding,
    LocalEmulator,
    AutoLocalEmulatorBackend,
)

def test_default_config() -> None:
    default_config = solvers.Config()
    check.equal(default_config.config_name, "")
    check.equal(default_config.solving_mode, "quantum")
    assert isinstance(default_config.solving, solvers.quantum.Config)
    check.is_instance(default_config.solving.backend, LocalEmulator)
    check.is_(default_config.solving.backend._backend_type, AutoLocalEmulatorBackend)
    check.equal(default_config.solving.embedding.algorithm, embedding.Algorithm.GREEDY_LAYOUT)
    check.equal(default_config.solving.embedding.greedy_layout_lattice, embedding.Lattice.TRIANGULAR)
    check.is_none(default_config.decompose)


def test_default_classical_config() -> None:
    default_classical = solvers.classical.Config()
    check.equal(default_classical.algorithm, solvers.classical.Algorithm.TABU_SEARCH)
    with pytest.raises(TypeError):
        solvers.classical.Config(algorithm=1)  # type: ignore[arg-type]


def test_drive_shaping_config() -> None:
    default_drive_shaping_config = drive_shaping.Config()
    check.equal(default_drive_shaping_config.algorithm, drive_shaping.Algorithm.PROPORTIONAL_DIAGONAL)

    check.equal(len(default_drive_shaping_config.bayesian_search_initial_detuning_parameters), 3)
    check.equal(len(default_drive_shaping_config.bayesian_search_initial_omega_parameters), 3)

    with pytest.raises(ValueError):
        drive_shaping.Config(algorithm="dummy")

    check.equal(
        drive_shaping.Config(algorithm="proportional_diagonal").algorithm,
        drive_shaping.Algorithm.PROPORTIONAL_DIAGONAL,
    )
    check.equal(
        drive_shaping.Config(algorithm="bayesian_search").algorithm,
        drive_shaping.Algorithm.BAYESIAN_SEARCH,
    )


def test_embdedding_config() -> None:
    default_embedding_config = embedding.Config()
    check.equal(default_embedding_config.algorithm, embedding.Algorithm.GREEDY_LAYOUT)
    check.equal(default_embedding_config.greedy_layout_lattice, embedding.Lattice.TRIANGULAR)
    check.is_true(default_embedding_config.greedy_layout_traps)

    with pytest.raises(ValueError):
        embedding.Config(algorithm="dummy")
    with pytest.raises(ValueError):
        embedding.Config(greedy_layout_lattice="dummy")  # type: ignore[arg-type]


def test_config_name() -> None:
    name_config = solvers.Config(config_name="my_config")
    check.equal(name_config.config_name, "my_config")


def test_classical_config_flag() -> None:
    classical_solver_config = solvers.Config(solving=solvers.classical.Config())
    check.equal(classical_solver_config.solving_mode, "classical")


def test_solving_mode() -> None:
    quantum_solver_config = solvers.Config(solving=solvers.quantum.Config())
    check.equal(quantum_solver_config.solving_mode, "quantum")

    classical_solver_config = solvers.Config(solving=solvers.classical.Config())
    check.equal(classical_solver_config.solving_mode, "classical")


def test_solving_mode_non_default_config() -> None:
    classical_solver_config = solvers.Config(
        solving=solvers.classical.Config(algorithm="simulated_annealing", max_iter=500),
    )
    check.equal(classical_solver_config.solving_mode, "classical")

    quantum_solver_config = solvers.Config(
        solving=solvers.quantum.Config(
            embedding=embedding.Config(algorithm="blade", blade_dimensions=[2]),
        ),
    )
    check.equal(quantum_solver_config.solving_mode, "quantum")


def test_quantum_config_property() -> None:
    quantum_solver_config = solvers.Config(solving=solvers.quantum.Config())
    check.is_(quantum_solver_config.quantum, quantum_solver_config.solving)

    classical_solver_config = solvers.Config(solving=solvers.classical.Config())
    with pytest.raises(ValueError):
        classical_solver_config.quantum


def test_classical_config_property() -> None:
    classical_solver_config = solvers.Config(solving=solvers.classical.Config())
    check.is_(classical_solver_config.classical, classical_solver_config.solving)

    quantum_solver_config = solvers.Config(solving=solvers.quantum.Config())
    with pytest.raises(ValueError):
        quantum_solver_config.classical


def test_qutip_config_backend() -> None:
    qutip_solver_config = solvers.Config(
        solving=solvers.quantum.Config(
            backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
        ),
    )
    assert isinstance(qutip_solver_config.solving, solvers.quantum.Config)
    check.is_(qutip_solver_config.solving.backend._backend_type, QutipBackendV2)


def test_blade_config() -> None:
    embed_method = embedding.Config(algorithm="blade", blade_dimensions=[2])
    blade_config = solvers.quantum.Config(embedding=embed_method)
    check.equal(blade_config.embedding.algorithm, embedding.Algorithm.BLADE)
    check.equal(type(blade_config.device), AnalogDeviceWithDMM)
    check.equal(blade_config.embedding.blade_dimensions, [2])


def test_blade_clear_dimensions_config() -> None:
    embed_method = embedding.Config(blade_dimensions=[6, 5, 4, 3, 2])
    blade_clear_dimensions_config = solvers.quantum.Config(embedding=embed_method)
    check.equal(blade_clear_dimensions_config.embedding.blade_dimensions, [6, 5, 4, 3, 2])


def test_greedy_embedding_config() -> None:

    embedding_config = embedding.Config(
        algorithm="greedy_layout",
        greedy_layout_lattice=embedding.Lattice.SQUARE,
        greedy_layout_traps=10,
    )
    config = solvers.quantum.Config(
        embedding=embedding_config,
    )
    check.equal(config.embedding.algorithm, embedding.Algorithm.GREEDY_LAYOUT)
    check.is_instance(config.device, AnalogDeviceWithDMM)
    check.equal(config.embedding.greedy_layout_lattice, embedding.Lattice.SQUARE)
    check.equal(config.embedding.greedy_layout_traps, 10)


def test_initialization_device() -> None:

    solver = solvers.quantum.Config()
    check.equal(solver.embedding.greedy_layout_traps, "device")


def test_decomposition_config() -> None:
    config = solvers.Config(decompose=solvers.DecompositionConfig())
    check.is_not_none(config.decompose)
