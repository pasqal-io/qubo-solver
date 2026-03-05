from __future__ import annotations

import pytest
import pytest_check as check
import numpy as np
import torch
from typing import Any, Dict, Sequence, Mapping
import io

from pulser.backend.remote import (
    BatchStatus,
    JobStatus,
    RemoteConnection,
    RemoteResults,
)
from pulser.result import Result, Results
from pulser_pasqal import PasqalCloud

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


class _MockConnection(PasqalCloud):
    def __init__(self, local_result: Result) -> None:
        self.result = local_result
        self.result.bitstring_counts = self.result.final_bitstrings  # type: ignore[attr-defined]
        self.results: Dict[str, Result] = dict()

        self._status_calls = 0
        self._support_open_batch = True
        self._got_closed = ""
        self._progress_calls = 0

    def submit(  # type: ignore[override]
        self,
        sequence: Sequence[Any],
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs: Any,
    ) -> RemoteResults:
        if not batch_id:
            batch_id = "abcd"
        self.results[batch_id] = self.result

        return RemoteResults(batch_id, self)

    def _fetch_result(self, batch_id: str, job_ids: list[str] | None = None) -> tuple[Results, ...]:
        return (self.results[batch_id],)

    def _query_job_progress(self, batch_id: str) -> Mapping[str, tuple[JobStatus, Result | None]]:
        if batch_id not in self.results.keys():
            return {batch_id: (JobStatus.ERROR, None)}
        return {batch_id: (JobStatus.DONE, self.results[batch_id])}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        if batch_id not in self.results.keys():
            return BatchStatus.ERROR
        return BatchStatus.DONE

    def _close_batch(self, batch_id: str) -> None:
        self._got_closed = batch_id

    def supports_open_batch(self) -> bool:
        return bool(self._support_open_batch)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("embedding_method", list(EmbedderType))
@pytest.mark.parametrize("preprocessing", [True, False])
@pytest.mark.parametrize("dmm", [True, False])
def test_quantum_batch_id(
    drive_method: str,
    embedding_method: str,
    preprocessing: bool,
    dmm: bool,
) -> None:
    if embedding_method is EmbedderType.BLADE:
        pytest.skip(reason="Blade embedding still has bugs")
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
    ) -> tuple[RemoteResults | Sequence[Results], QuboSolverQuantum]:
        instance = QUBOInstance(Q)

        config = SolverConfig(use_quantum=True, do_preprocessing=preprocessing)
        config.embedding = EmbeddingConfig(
            embedding_method=embedding_method, greedy_spacing=7.0, greedy_traps=100
        )
        config.drive_shaping = DriveShapingConfig(drive_shaping_method=drive_method, dmm=dmm)
        runs = 50

        if connection is None:
            config.backend = LocalEmulator(runs=runs)
            wait = True
        else:
            config.backend = RemoteEmulator(connection=connection, runs=runs)
            wait = False

        solver = QuboSolverQuantum(instance, config)

        solver._check_size_limit()

        # 2) Apply preprocessing if requested
        solver.preprocess()

        embedding = solver.embedding()
        drive, _ = solver.drive(embedding)
        results = solver.submit(drive, embedding, wait)

        return results, solver

    def post(results: RemoteResults | Sequence[Results], solver: QuboSolverQuantum) -> QUBOSolution:
        bitstrings, counts = QuboSolverQuantum.parse_results(results)

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

    local_results, local_solver = pre()
    local_solution = post(local_results, local_solver)

    connection = _MockConnection(local_results[0])
    remote_results, remote_solver = pre(connection)
    assert isinstance(remote_results, RemoteResults)

    mock_file = io.BytesIO()
    QuboSolverQuantum.save(mock_file, remote_solver)
    io_utils.save_string(mock_file, remote_results.batch_id)

    with pytest.raises(RuntimeError):
        _ = QuboSolverQuantum.get_results("invalid_batch_id", connection)
    remote_results_invalid = QuboSolverQuantum.get_results(
        "invalid_batch_id", connection, check=False
    )
    assert remote_results_invalid is not None
    check.equal(remote_results_invalid.get_batch_status().name, "ERROR")

    mock_file.seek(0)
    remote_solver_2 = QuboSolverQuantum.load(mock_file)
    batch_id_2 = io_utils.load_string(mock_file)
    remote_results_2 = QuboSolverQuantum.get_results(batch_id_2, connection)
    assert remote_results_2 is not None
    check.equal(remote_results_2.get_batch_status().name, "DONE")
    remote_solution = post(remote_results_2, remote_solver_2)

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
