from __future__ import annotations

import numpy as np
import pytest

import io
import pytest_check as check
import torch
from unittest.mock import Mock

from qoolqit import Device
from qoolqit.execution import JobStatus

from qubosolver import (
    Instance,
    Solver,
    QuboSolver,
    Solution,
    EmbedderType,
    Analyzer,
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
    transforms,
    vectori,
    bitstrings,
    matrix,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.solvers.solver import _QuboSolverQuantum
from mock.connection import MockConnection

from pulser.backend.remote import (
    Results,
    RemoteConnection,
)
from emu_sv import SVBackend

from typing import Optional


@pytest.fixture
def implicit_default_qubo_solver_config(
    simple_qubo_instance: Instance,
) -> Solver:
    default_solver = Solver(simple_qubo_instance)
    return default_solver


def test_implicit_solver_config(
    implicit_default_qubo_solver_config: Solver,
) -> None:
    assert isinstance(implicit_default_qubo_solver_config._solver, _QuboSolverQuantum)


def test_different_shots(simple_qubo_instance: Instance) -> None:
    from pulser_simulation import QutipBackendV2

    default_solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500)
        ),
    )
    solutions = default_solver.solve()
    assert solutions.counts.sum() == 500

    lessshots_solver = Solver(
        simple_qubo_instance,
        SolverConfig(
            use_quantum=True, backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=100)
        ),
    )
    solutions = lessshots_solver.solve()
    assert solutions.counts.sum() == 100


@pytest.mark.priority(40)
@pytest.mark.flaky(max_runs=5)
def test_run_local_backends(simple_qubo_instance: Instance, local_backend: LocalEmulator) -> None:
    solver = Solver(
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


@pytest.mark.priority(150)
def test_solver_different_devices(
    request: pytest.FixtureRequest,
    qubo_for_testing_many_devices: Instance,
    local_device: Device,
    embedding_method: EmbedderType,
) -> None:

    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method="heuristic"),
        embedding=EmbeddingConfig(
            embedding_method=embedding_method,
            greedy_traps=qubo_for_testing_many_devices.size,
            min_distance=1.001,
        ),
        do_postprocessing=False,
        do_preprocessing=False,
        device=local_device,
        backend=LocalEmulator(backend_type=SVBackend),
    )
    solver = Solver(qubo_for_testing_many_devices, config)
    solution = solver.solve()
    assert solution


def test_parse_results() -> None:
    # Mock results object
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"001": 10, "110": 5, "010": 3}

    solution = Solution.from_results(mock_result)

    expected_bitstrings = bitstrings.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    expected_counts = vectori.tensor([10, 5, 3])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_empty_final_bitstrings() -> None:
    # Mock results with empty final_bitstrings
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {}

    solution = Solution.from_results(mock_result)

    check.equal(solution.bitstrings.shape, (0, 0))
    check.equal(solution.bitstrings.dtype, torch.int8)
    check.equal(solution.counts.shape, (0,))
    check.equal(solution.counts.dtype, torch.int64)


def test_parse_results_binary_string_conversion() -> None:
    # Mock results with binary string keys
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"0101": 8, "1010": 12, "1111": 4}

    solution = Solution.from_results(mock_result)

    expected_bitstrings = bitstrings.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    expected_counts = vectori.tensor([8, 12, 4])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_single_bitstring() -> None:
    # Mock results with single bitstring
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": 25}

    solution = Solution.from_results(mock_result)

    expected_bitstrings = bitstrings.tensor([[1, 0, 1]])
    expected_counts = vectori.tensor([25])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def test_parse_results_string_counts_to_integer_tensor() -> None:
    # Mock results with string count values
    mock_result = Mock(spec=Results)
    mock_result.final_bitstrings = {"101": "15", "010": "8", "111": "12"}

    solution = Solution.from_results(mock_result)

    expected_bitstrings = bitstrings.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    expected_counts = vectori.tensor([15, 8, 12])

    torch.testing.assert_close(solution.bitstrings, expected_bitstrings)
    torch.testing.assert_close(solution.counts, expected_counts)


def trivial_triangular_qubo(connection: Optional[RemoteConnection] = None) -> Solver:
    Q = 10.0 * matrix.tensor(
        [
            [-10.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    qubo = Instance(Q)

    config = SolverConfig(use_quantum=True, do_preprocessing=False)

    config.embedding = EmbeddingConfig(embedding_method="blade")
    num_shots = 100

    if connection is None:
        config.backend = LocalEmulator(num_shots=num_shots)
    else:
        config.backend = RemoteEmulator(connection=connection, num_shots=num_shots)

    solver = Solver(qubo, config)

    return solver


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("wait", [True, False], ids=["wait", "dont_wait"])
def test_submit_integration(make_mock_connection: type[MockConnection], wait: bool) -> None:
    seed = 16844214
    np.random.seed(seed)

    solver = trivial_triangular_qubo()

    embedding = solver.embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver.drive(embedding)

    job = solver.submit(drive, embedding)
    results = job.results()

    solution = Solution.from_results(results)
    solution.compute_costs(solver.instance.matrix).compute_probabilities()

    # Take the top 3 solutions with the highest probabilities
    sorted_indices = torch.argsort(solution.probabilities, descending=True)
    bitstrings_ = solution.bitstrings[sorted_indices][0:3, :]
    # Sort them by lexicographic order
    np_sorted_indices = np.lexsort(bitstrings_.numpy().T[::-1])
    bitstrings_ = bitstrings_[np_sorted_indices, :]

    torch.testing.assert_close(bitstrings_, bitstrings.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

    analyzer = Analyzer([solution])
    print(f"\n{analyzer.df}")

    # Remote solutions should be identical to local ones
    np.random.seed(seed)

    assert isinstance(results, Results)
    solver_remote = trivial_triangular_qubo(make_mock_connection(results, running_iterations=1))

    embedding = solver_remote.embedding()
    # Qoolqit's embedding has an hardcoded seed. Set the seed ourselves.
    np.random.seed(seed)
    drive, _ = solver_remote.drive(embedding)
    remote_job = solver_remote.submit(drive, embedding)

    if not wait:
        with pytest.raises(TimeoutError):
            remote_job.results(0)

    results_remote = remote_job.results()
    assert isinstance(results_remote, Results)
    check.equal(remote_job.get_status(), JobStatus.DONE)

    solution_remote = Solution.from_results(results)
    torch.testing.assert_close(solution_remote.bitstrings, solution.bitstrings)
    torch.testing.assert_close(solution_remote.counts, solution.counts)


@pytest.mark.parametrize("postprocessing", [True, False], ids=["post", "no-post"])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no-pre"])
def test_save_load_qubo_solver_quantum(
    postprocessing: bool,
    preprocessing: bool,
) -> None:

    Q = matrix.tensor(
        [
            [0.0, 6.0, 6.0],
            [6.0, -10.0, 6.0],
            [6.0, 6.0, -10.0],
        ]
    )
    expected_preprocessed_Q = matrix.tensor(
        [
            [-10.0, 6.0],
            [6.0, -10.0],
        ]
    )
    qubo = Instance(Q)
    config = SolverConfig(do_preprocessing=preprocessing, do_postprocessing=postprocessing)
    solver = Solver(qubo, config)
    solver.preprocess()

    if preprocessing:
        assert isinstance(solver.instance, transforms.variable_fixing.Instance)
        torch.testing.assert_close(solver.instance.matrix, expected_preprocessed_Q)
        torch.testing.assert_close(solver.instance._parent_instance.matrix, Q)
    else:
        check.is_not_instance(solver.instance, transforms.variable_fixing.Instance)
        torch.testing.assert_close(solver.instance.matrix, Q)

    # Save the solver
    file = io.BytesIO()
    Solver.save(file, solver)

    # Load the solver
    file.seek(0)
    loaded_solver = Solver.load(file)

    # Verify the loaded solver has the same properties
    torch.testing.assert_close(loaded_solver.instance.matrix, solver.instance.matrix)
    if preprocessing:
        assert isinstance(solver.instance, transforms.variable_fixing.Instance)
        assert isinstance(loaded_solver.instance, transforms.variable_fixing.Instance)
        torch.testing.assert_close(
            loaded_solver.instance._parent_instance.matrix,
            solver.instance._parent_instance.matrix,
        )
        check.equal(loaded_solver.instance._fixed_indices, solver.instance._fixed_indices)
    else:
        check.is_not_instance(loaded_solver.instance, transforms.variable_fixing.Instance)
    check.equal(loaded_solver.config.do_preprocessing, solver.config.do_preprocessing)
    check.equal(loaded_solver.config.do_postprocessing, solver.config.do_postprocessing)

    for method in [
        "solve",
        "embedding",
        "drive",
        "submit",
        "execute",
        "draw_sequence",
        "preprocess",
        "_trivial_solution",
    ]:
        with pytest.raises(
            AttributeError,
            match=f"'{method}' is disabled: this method is not supported for QuboSolverQuantum loaded from a file.",
        ):
            getattr(loaded_solver, method)()


def test_qubo_solver_wrong_case_deprecation() -> None:
    Q = matrix.tensor([[-2.0, 0.0], [0.0, 4.0]])
    with pytest.warns(DeprecationWarning, match="Use `qubosolver.Solver` instead"):
        solver = QuboSolver(Instance(Q))
        check.is_instance(solver, Solver)
    solution = solver.solve()
    check.equal(solution[0].string, "10")
