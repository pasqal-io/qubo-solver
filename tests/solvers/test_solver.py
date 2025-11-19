from __future__ import annotations

import pytest
from qoolqit.devices import Device
from qubosolver.config import EmbeddingConfig, DriveShapingConfig, SolverConfig, LocalEmulator
from qubosolver.qubo_types import EmbedderType
from qubosolver.solver import QUBOInstance, QuboSolver, QuboSolverClassical


@pytest.fixture
def implicit_default_qubo_solver_config(
    simple_qubo_instance: QUBOInstance,
) -> QuboSolver:
    default_solver = QuboSolver(simple_qubo_instance)
    return default_solver


def test_implicit_solver_config(
    implicit_default_qubo_solver_config: QuboSolver,
) -> None:
    assert isinstance(implicit_default_qubo_solver_config._solver, QuboSolverClassical)


def test_different_shots(simple_qubo_instance: QUBOInstance) -> None:
    from pulser_simulation import QutipBackendV2

    default_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, runs=500)
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500  # type: ignore[union-attr]

    lessshots_solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, runs=100)
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100  # type: ignore[union-attr]


@pytest.mark.flaky(reruns=5)
def test_run_local_backends(
    simple_qubo_instance: QUBOInstance, local_backend: LocalEmulator
) -> None:
    solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True,
            backend=local_backend,
            embedding=EmbeddingConfig(embedding_method=EmbedderType.BLADE),
        ),
    )
    solutions = solver.solve()
    # theoretically -4.4000 can be found
    assert solutions.costs.min().item() <= -3.0


@pytest.mark.parametrize(
    "drive_shaping_method",
    [
        "adiabatic",
    ],
)
def test_solver_different_devices_drives(
    qubo_instance_for_embedding: QUBOInstance,
    local_device: Device,
    local_backend: LocalEmulator,
    drive_shaping_method: str,
) -> None:
    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=drive_shaping_method),
        do_postprocessing=False,
        do_preprocessing=False,
        backend=local_backend,
        device=local_device,
    )
    solver = QuboSolver(qubo_instance_for_embedding, config)
    solution = solver.solve()
    assert solution
