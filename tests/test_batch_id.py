from __future__ import annotations

import pytest
import pytest_check as check
import numpy as np
import torch
import io

from pulser.backend.remote import RemoteConnection
from pulser.backend.results import Results

from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.config import (
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.qubo_types import EmbedderType, DriveType
from qubosolver.solver import QUBOInstance, QuboSolverQuantum, QUBOSolution
import qubosolver.io.utils as io_utils

from qoolqit import DigitalAnalogDevice
from qoolqit.execution import retrieve_remote_job, get_batch_id, job, JobStatus
from mock.connection import MockConnection


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("embedding_method", list(EmbedderType))
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no_pre"])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_quantum_batch_id(
    make_mock_connection: type[MockConnection],
    drive_method: str,
    embedding_method: str,
    preprocessing: bool,
    dmm: bool,
) -> None:
    if drive_method == DriveType.OPTIMIZED:
        pytest.skip(reason="Does not work with the optimized drive shaping method")

    np.random.seed(7979)

    Q = np.array(
        [
            [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ]
    )

    def pre(
        connection: RemoteConnection | None = None,
    ) -> tuple[job.Job[Results], QuboSolverQuantum]:
        instance = QUBOInstance(Q)

        min_distance = 1.001 if drive_method == DriveType.HEURISTIC else None

        config = SolverConfig(
            use_quantum=True, do_preprocessing=preprocessing, device=DigitalAnalogDevice()
        )
        config.embedding = EmbeddingConfig(
            embedding_method=embedding_method,
            greedy_spacing=7.0,
            greedy_traps=100,
            min_distance=min_distance,
        )
        config.drive_shaping = DriveShapingConfig(drive_shaping_method=drive_method, dmm=dmm)
        num_shots = 50

        if connection is None:
            config.backend = LocalEmulator(num_shots=num_shots)
        else:
            config.backend = RemoteEmulator(connection=connection, num_shots=num_shots)

        solver = QuboSolverQuantum(instance, config)

        solver._check_size_limit()

        # 2) Apply preprocessing if requested
        solver.preprocess()

        embedding = solver.embedding()
        drive, _ = solver.drive(embedding)
        job = solver.submit(drive, embedding)

        return job, solver

    def post(job: job.Job[Results], solver: QuboSolverQuantum) -> QUBOSolution:
        bitstrings, counts = QuboSolverQuantum.parse_results(job.results())

        solution = QUBOSolution(
            bitstrings=bitstrings.float(),
            counts=counts,
            costs=torch.Tensor(),
            probabilities=None,
        )

        # Post-process fixations of the preprocessing and restore the original QUBO
        solution = solver.post_process_fixation(solution)
        solution = solver.post_process(solution)

        solution.costs = solution.compute_costs(solver.instance)
        solution.probabilities = solution.compute_probabilities()
        solution.sort_by_cost()

        solution.bitstrings = solution.bitstrings.int()

        return solution

    local_job, local_solver = pre()
    local_solution = post(local_job, local_solver)

    connection = make_mock_connection(local_job.results())
    remote_job, remote_solver = pre(connection)
    assert isinstance(remote_job.results(), Results)

    mock_file = io.BytesIO()
    QuboSolverQuantum.save(mock_file, remote_solver)
    io_utils.save_string(mock_file, remote_job.job_id())
    io_utils.save_string(mock_file, get_batch_id(remote_job))

    with pytest.raises(ValueError):
        invalid_job = retrieve_remote_job(connection, "invalid_job_id", batch_id="invalid_batch_id")
        invalid_job.get_status()

    mock_file.seek(0)
    remote_solver_2 = QuboSolverQuantum.load(mock_file)
    job_id_2 = io_utils.load_string(mock_file)
    batch_id_2 = io_utils.load_string(mock_file)
    remote_job_2 = retrieve_remote_job(connection, job_id_2, batch_id=batch_id_2)
    check.equal(remote_job_2.get_status(), JobStatus.DONE)
    remote_solution = post(remote_job_2, remote_solver_2)

    torch.testing.assert_close(remote_solution.bitstrings, local_solution.bitstrings)
    torch.testing.assert_close(remote_solution.costs, local_solution.costs)
    torch.testing.assert_close(remote_solution.probabilities, local_solution.probabilities)
    torch.testing.assert_close(remote_solution.counts, local_solution.counts)

    analyzer = QUBOAnalyzer([local_solution, remote_solution], labels=["local", "remote"])
    print(f"\n{analyzer.df}")

    expected_solutions = ["00111", "01011"]

    for label in ["local", "remote"]:
        df = analyzer.df.query(f"labels == '{label}'")

        check.is_true(df["bitstrings"].is_unique)

        probabilities = [
            df.set_index("bitstrings")["probs"].get(b, 0.0) for b in expected_solutions
        ]
        check.greater(max(probabilities), 0.4)

        for b in expected_solutions:
            if b in df["bitstrings"].values:
                cost = df.set_index("bitstrings")["costs"].get(b)
                np.testing.assert_allclose(cost, -27.288260)
