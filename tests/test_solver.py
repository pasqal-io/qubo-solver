from __future__ import annotations

import pytest
import torch
from qoolqit._solvers.data import BackendConfig

from qubosolver.config import EmbeddingConfig, SolverConfig
from qubosolver.qubo_types import EmbedderType
from qubosolver.solver import QUBOInstance, QuboSolver, QuboSolverClassical


@pytest.fixture
def simple_qubo_instance() -> QUBOInstance:
    coefficients = torch.tensor([[-1.0, 0.5, 0.2], [0.5, -2.0, 0.3], [0.2, 0.3, -3.0]])
    return QUBOInstance(coefficients=coefficients)


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

    default_solver = QuboSolver(simple_qubo_instance, SolverConfig(use_quantum=True))
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500  # type: ignore[union-attr]

    lessshots_solver = QuboSolver(
        simple_qubo_instance, SolverConfig(use_quantum=True, num_shots=100)
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100  # type: ignore[union-attr]


def test_run_local_backends(
    simple_qubo_instance: QUBOInstance, local_backend: BackendConfig
) -> None:

    solver = QuboSolver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True,
            backend_config=local_backend,
        ),
    )
    solutions = solver.solve()
    assert solutions.costs.min().detach().item() < 0
