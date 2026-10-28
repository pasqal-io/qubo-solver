# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import pytest_check as check
import qoolqit
import torch
from pulser.backend.remote import (
    RemoteConnection,
    Results,
)
from pulser_simulation import QutipBackendV2
from qoolqit.execution import JobStatus
from qoolqit.graphs import DataGraph
from scipy.spatial.distance import pdist, squareform

from qubosolver import (
    ClassicalSolvingConfig,
    DriveShapingConfig,
    EmbeddingConfig,
    Instance,
    LocalEmulator,
    QuantumSolvingConfig,
    RemoteEmulator,
    Solution,
    Solver,
    SolverConfig,
    bitstrings,
    matrix,
    vectori,
)
from qubosolver.solver.solver import _QuboSolverClassical, _QuboSolverQuantum
from qubosolver.utils import _random, analysis

if TYPE_CHECKING:
    from mock.connection import MockConnection
    from qoolqit import Device

    from qubosolver.solver.config.embedding import _EmbeddingAlgorithm


@pytest.fixture
def implicit_default_qubo_solver_config(
    simple_qubo_instance: Instance,
) -> Solver:
    return Solver(simple_qubo_instance)


def test_implicit_solver_config(
    implicit_default_qubo_solver_config: Solver,
) -> None:
    assert isinstance(implicit_default_qubo_solver_config._solver, _QuboSolverQuantum)


@pytest.mark.duration(21)
def test_different_shots(simple_qubo_instance: Instance) -> None:
    default_solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            solving=QuantumSolvingConfig(
                backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500)
            ),
            preprocessing=False,
            postprocessing=False,
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500

    lessshots_solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            solving=QuantumSolvingConfig(
                backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=100)
            ),
            preprocessing=False,
            postprocessing=False,
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100


@pytest.mark.duration(40)
@pytest.mark.flaky(reruns=5)
def test_run_local_backends(simple_qubo_instance: Instance, local_backend: LocalEmulator) -> None:
    solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            solving=QuantumSolvingConfig(
                backend=local_backend,
                embedding=EmbeddingConfig(algorithm="blade"),
            ),
            preprocessing=False,
            postprocessing=False,
        ),
    )
    # simple_qubo_instance is tiny, so non-QutipBackendV2 backends are intentionally
    # suboptimal here and expected to warn.
    if local_backend._backend_type is QutipBackendV2:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solutions = solver.solve()
    else:
        with pytest.warns(UserWarning, match="Using .* Consider using"):
            solutions = solver.solve()
    # theoretically -4.4000 can be found
    assert solutions.costs.min().item() <= -3.0


@pytest.mark.duration(150)
def test_solver_different_devices(
    qubo_for_testing_many_devices: Instance,
    local_device: Device,
    embedding_algorithm: _EmbeddingAlgorithm,
) -> None:

    quantum_config = QuantumSolvingConfig(
        drive_shaping=DriveShapingConfig(algorithm="proportional_diagonal"),
        embedding=EmbeddingConfig(
            algorithm=embedding_algorithm,
        ),
        device=local_device,
        backend=LocalEmulator(),
    )
    config = SolverConfig(
        solving=quantum_config,
        postprocessing=False,
        preprocessing=False,
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    solution = solver.solve()
    assert solution


def test_parse_results() -> None:
    # Mock results object
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"001": 10, "110": 5, "010": 3}

    solution = Solution.from_results(mock_result, Instance(matrix.zeros(3)))

    expected_bitstrings = bitstrings.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    expected_counts = vectori.tensor([10, 5, 3])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_empty_final_bitstrings() -> None:
    # Mock results with empty final_bitstrings
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {}

    solution = Solution.from_results(mock_result, Instance())

    check.equal(solution.bitstrings.shape, (0, 0))
    check.equal(solution.bitstrings.dtype, torch.int8)
    check.equal(solution.counts.shape, (0,))
    check.equal(solution.counts.dtype, torch.int64)


def test_parse_results_binary_string_conversion() -> None:
    # Mock results with binary string keys
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"0101": 8, "1010": 12, "1111": 4}

    solution = Solution.from_results(mock_result, Instance(matrix.zeros(4)))

    expected_bitstrings = bitstrings.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    expected_counts = vectori.tensor([8, 12, 4])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_single_bitstring() -> None:
    # Mock results with single bitstring
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": 25}

    solution = Solution.from_results(mock_result, Instance(matrix.zeros(3)))

    expected_bitstrings = bitstrings.tensor([[1, 0, 1]])
    expected_counts = vectori.tensor([25])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_string_counts_to_integer_tensor() -> None:
    # Mock results with string count values
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": "15", "010": "8", "111": "12"}

    solution = Solution.from_results(mock_result, Instance(matrix.zeros(3)))

    expected_bitstrings = bitstrings.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    expected_counts = vectori.tensor([15, 8, 12])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def trivial_triangular_qubo(connection: RemoteConnection | None = None) -> Solver:
    Q = 10.0 * matrix.tensor(
        [
            [-10.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    qubo = Instance(Q)

    num_shots = 100
    backend = (
        LocalEmulator(num_shots=num_shots)
        if connection is None
        else RemoteEmulator(connection=connection, num_shots=num_shots)
    )
    config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm="blade"),
            backend=backend,
        ),
        preprocessing=False,
    )

    return Solver(qubo, config)


@pytest.mark.duration(25)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("wait", [True, False], ids=["wait", "dont_wait"])
def test_submit_integration(make_mock_connection: type[MockConnection], wait: bool) -> None:
    seed = 16844214
    np.random.seed(seed)

    solver = trivial_triangular_qubo()

    embedding = solver._embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver._drive(embedding)

    job = solver._submit(drive, embedding)
    results = job.results()

    solution = Solution.from_results(results, solver.instance)

    # Take the top 3 solutions with the highest probabilities
    sorted_indices = torch.argsort(solution.probabilities, descending=True)
    bitstrings_ = solution.bitstrings[sorted_indices][0:3, :]
    # Sort them by lexicographic order
    np_sorted_indices = np.lexsort(bitstrings_.numpy().T[::-1])
    bitstrings_ = bitstrings_[np_sorted_indices, :]

    torch.testing.assert_close(bitstrings_, bitstrings.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

    print(f"\n{analysis.to_dataframe([solution])}")

    # Remote solutions should be identical to local ones
    np.random.seed(seed)

    assert isinstance(results, Results)
    solver_remote = trivial_triangular_qubo(make_mock_connection(results, running_iterations=1))

    embedding = solver_remote._embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver_remote._drive(embedding)
    remote_job = solver_remote._submit(drive, embedding)

    if not wait:
        with pytest.raises(TimeoutError):
            remote_job.results(0)

    results_remote = remote_job.results()
    assert isinstance(results_remote, Results)
    check.equal(remote_job.get_status(), JobStatus.DONE)

    solution_remote = Solution.from_results(results, solver.instance)
    torch.testing.assert_close(solution_remote.bitstrings, solution.bitstrings)
    torch.testing.assert_close(solution_remote.counts, solution.counts)


@pytest.mark.duration(19)
def test_respects_total_bottom_detuning(caplog: pytest.LogCaptureFixture) -> None:
    n = 6
    Q = matrix.zeros(n)
    for i in range(n):
        Q[i, i] = -50.0 if i % 2 == 0 else 50.0
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = Q[j, i] = 1.0

    instance = Instance(Q)
    config = SolverConfig(
        solving=QuantumSolvingConfig(drive_shaping=DriveShapingConfig()),
        preprocessing=False,
        postprocessing=False,
    )

    with caplog.at_level(logging.INFO):
        solution = Solver(instance, config).solve()

    check.is_in("DMM final detuning would exceed the device's total_bottom_detuning", caplog.text)
    check.is_true(solution)


def _triangular_register_qubo() -> Instance:
    data_graph = DataGraph.triangular(4, 4, 1)
    rng = np.random.default_rng(0)
    removed = rng.choice(
        data_graph.number_of_nodes(), data_graph.number_of_nodes() - 8, replace=False
    )
    data_graph.remove_nodes_from(removed)
    coords = [data_graph.coords[n] for n in data_graph.nodes]
    dist_matrix = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        qubo = 1.0 / dist_matrix**6
    np.fill_diagonal(qubo, -0.5)
    return Instance(matrix.as_tensor(np.asarray(qubo, dtype=float)))


@pytest.mark.duration(4)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("algorithm", ["greedy_layout", "blade"])
def test_quantum_matches_classical_triangular(algorithm: _EmbeddingAlgorithm) -> None:
    _random.manual_seed(4425)
    instance = _triangular_register_qubo()

    quantum_config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm=algorithm),
            drive_shaping=DriveShapingConfig(algorithm="proportional_diagonal"),
        ),
        preprocessing=False,
        postprocessing=False,
    )
    quantum_solution = Solver(instance, quantum_config).solve()

    classical_config = SolverConfig(
        solving=ClassicalSolvingConfig(max_bitstrings=100),
    )
    classical_solution = Solver(instance, classical_config).solve()

    check.almost_equal(
        quantum_solution[0].cost,
        classical_solution[0].cost,
    )

    classical_solution.check_consistency(instance=instance, throw=True)
    quantum_solution.check_consistency(instance=instance, throw=True)

    check.greater_equal(quantum_solution[0].probability, 0.1)

    print()
    print(
        analysis.to_dataframe(
            [classical_solution, quantum_solution], labels=["classical", "quantum"]
        )
    )


def test_quantum_solver_skips_pipeline_when_preprocessing_fixes_every_variable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Not caught by trivial_solution_search (mixed-sign, non-diagonal), but
    # recursive Hansen fixing reduces it to zero variables, so the quantum
    # pipeline (embedding/drive/execute) must be skipped entirely.
    Q = matrix.tensor(
        [
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 1.0],
            [3.0, 1.0, -5.0],
        ]
    )
    instance = Instance(Q)
    config = SolverConfig(preprocessing=True, postprocessing=False)

    with caplog.at_level(logging.INFO):
        solution = Solver(instance, config).solve()

    check.is_in(
        "Instance has zero variables after preprocessing; skipping quantum pipeline.",
        caplog.text,
    )
    assert len(solution) == 1
    check.equal(len(solution[0].string), instance.size)
    check.almost_equal(solution[0].cost, -5.0)
    solution.check_consistency(instance=instance, throw=True)


def test_draw_sequence_draws_the_compiled_program(
    simple_qubo_instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    draw = Mock()
    monkeypatch.setattr(qoolqit.QuantumProgram, "draw", draw)
    solver = _QuboSolverQuantum(simple_qubo_instance, SolverConfig(preprocessing=False))
    embedding = solver._embedding()
    drive, _ = solver._drive(embedding)
    solver._draw_sequence(drive, embedding)
    draw.assert_called_once_with(compiled=True)

    # In classical mode no quantum program exists, so nothing is drawn.
    classical_solver = _QuboSolverClassical(
        simple_qubo_instance, SolverConfig(solving=ClassicalSolvingConfig())
    )
    classical_solver._draw_sequence(drive, embedding)
    draw.assert_called_once()


def test_update_instance_updates_inner_solver(
    simple_qubo_instance: Instance, simple_qubo_instance2: Instance
) -> None:
    solver = Solver(simple_qubo_instance)
    solver._update_instance(simple_qubo_instance2)
    check.is_(solver.instance, simple_qubo_instance2)
    check.is_(solver._solver.instance, simple_qubo_instance2)


def test_negative_off_diagonal_without_preprocessing_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    instance = Instance(matrix.tensor([[-1.0, -1.0], [-1.0, -1.0]]))
    with caplog.at_level(logging.WARNING, logger="qubosolver.solver.solver"):
        Solver(instance, SolverConfig(preprocessing=False))
    check.is_in("Negative off-diagonal coefficients detected", caplog.text)


def test_classical_solver_has_no_embedding_nor_drive(simple_qubo_instance: Instance) -> None:
    solver = Solver(simple_qubo_instance, SolverConfig(solving=ClassicalSolvingConfig()))
    with pytest.raises(NotImplementedError):
        solver._embedding()
    with pytest.raises(NotImplementedError):
        solver._drive(qoolqit.Register.from_coordinates([[0.0, 0.0]]))
