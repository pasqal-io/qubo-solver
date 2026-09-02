from __future__ import annotations

import math
from typing import Callable
import torch

import pytest
import pytest_check as check

import pulser
from pulser.backend import EmulationConfig
from pulser.backend.default_observables import BitStrings
from pulser.backend.remote import BatchStatus, JobStatus, RemoteResults, RemoteResultsError
from pulser.backend.results import Results

import qoolqit
from qoolqit.execution import Job, JobStatus, retrieve_remote_job, get_batch_id

from qubosolver import Instance, RemoteEmulator, drive_shaping, matrix, vector, Solution, embedding, solving, analysis
from qubosolver.utils._local_connection import _QUBIT_LIMIT, LocalConnection

NUM_SHOTS = 50


def _program(*, n_qubits: int) -> qoolqit.QuantumProgram:
    """Build a compiled program of `n_qubits` atoms."""
    register = qoolqit.Register.circle(n_qubits)
    Q = matrix.as_tensor(register.interaction_matrix()) + torch.diag(vector.zeros(n_qubits).fill_(-1.0))
    instance = Instance(Q)
    device = qoolqit.AnalogDevice()
    drive = drive_shaping.proportional_diagonal.build_drive(
        instance, register, dmm=False, device=device
    )
    program = qoolqit.QuantumProgram(register, drive)
    program.compile_to(device, profile="max_energy", device_max_duration_ratio=0.999)
    return program


def _sequence(*, n_qubits: int) -> pulser.Sequence:
    """Compile a sequence of `n_qubits` atoms, cheap enough to emulate."""
    return _program(n_qubits=n_qubits).compiled_sequence


def _config() -> EmulationConfig:
    """Collect `NUM_SHOTS` bitstrings at the end of the sequence."""
    return EmulationConfig(
        observables=[BitStrings(num_shots=NUM_SHOTS)], default_evaluation_times=[1.0]
    )


def test_submit_returns_completed_bitstring_results() -> None:
    remote_results = LocalConnection().submit(
        _sequence(n_qubits=3), backend_configuration=_config()
    )

    check.equal(remote_results.get_batch_status(), BatchStatus.DONE)
    check.equal(len(remote_results.results), 1)

    results = remote_results.results[0]
    check.is_instance(results, Results)
    check.equal(results.get_result_tags(), ["bitstrings"])

    counts = results.get_result("bitstrings", 1.0)
    check.equal(sum(counts.values()), NUM_SHOTS)
    check.is_true(all(len(bitstring) == 3 for bitstring in counts))


def test_batch_holds_a_single_job() -> None:
    connection = LocalConnection()
    remote_results = connection.submit(_sequence(n_qubits=3), backend_configuration=_config())
    batch_id = remote_results.batch_id

    check.equal(remote_results.job_ids, [f"{batch_id}-0"])
    check.equal(connection._get_job_ids(batch_id), [f"{batch_id}-0"])


def test_query_job_progress_reports_done_with_results() -> None:
    connection = LocalConnection()
    remote_results = connection.submit(_sequence(n_qubits=3), backend_configuration=_config())
    progress = connection._query_job_progress(remote_results.batch_id)

    check.equal(list(progress), remote_results.job_ids)
    status, results = progress[remote_results.job_ids[0]]
    check.equal(status, JobStatus.DONE)
    check.is_instance(results, Results)
    check.equal(list(remote_results.get_available_results()), remote_results.job_ids)


def test_each_submission_creates_its_own_batch() -> None:
    connection = LocalConnection()
    sequence, config = _sequence(n_qubits=3), _config()

    first = connection.submit(sequence, backend_configuration=config)
    second = connection.submit(sequence, batch_id=first.batch_id, backend_configuration=config)

    # `batch_id` is ignored, so the second submission does not extend the first.
    check.not_equal(first.batch_id, second.batch_id)
    check.equal(len(connection._batches), 2)
    check.equal(len(second.results), 1)


def test_unknown_batch_reports_error_status() -> None:
    """A status query answers with a status rather than raising."""
    check.equal(LocalConnection()._get_batch_status("unknown"), BatchStatus.ERROR)


def test_job_params_are_ignored() -> None:
    """Document that `job_params` is dropped, unlike on a real connection.

    A real connection runs one job per `job_params` entry, each sampled
    `runs` times. Here the shot count comes from the emulation config only,
    and a batch always holds a single job. Reachable by driving a pulser
    remote backend directly, but not through [`RemoteEmulator`][], which
    never sets `job_params`.
    """
    remote_results = LocalConnection().submit(
        _sequence(n_qubits=3),
        job_params=[{"runs": 7}, {"runs": 7}],
        backend_configuration=_config(),
    )

    # `runs` is dropped: NUM_SHOTS from the config is used instead of 7.
    counts = remote_results.results[0].get_result("bitstrings", 1.0)
    check.equal(sum(counts.values()), NUM_SHOTS)

    # The second job is dropped too, rather than being executed.
    check.equal(len(remote_results.job_ids), 1)
    check.equal(len(remote_results.results), 1)


@pytest.mark.parametrize(
    "lookup",
    [
        lambda connection: connection._get_job_ids("unknown"),
        lambda connection: connection._fetch_result("unknown", None),
        lambda connection: connection._query_job_progress("unknown"),
        lambda connection: RemoteResults("unknown", connection, job_ids=["unknown-0"]),
    ],
    ids=["get_job_ids", "fetch_result", "query_job_progress", "remote_results"],
)
def test_unknown_batch_is_rejected(lookup: Callable[[LocalConnection], object]) -> None:
    """Every lookup rejects an unknown batch instead of inventing a job for it."""
    with pytest.raises(RemoteResultsError, match="Unknown batch 'unknown'"):
        lookup(LocalConnection())


def test_fetch_result_accepts_the_batch_own_job() -> None:
    connection = LocalConnection()
    remote_results = connection.submit(_sequence(n_qubits=3), backend_configuration=_config())
    batch_id = remote_results.batch_id

    check.equal(len(connection._fetch_result(batch_id, None)), 1)
    check.equal(len(connection._fetch_result(batch_id, connection._get_job_ids(batch_id))), 1)
    check.equal(len(RemoteResults(batch_id, connection, job_ids=remote_results.job_ids).results), 1)


@pytest.mark.parametrize("job_ids", [["unknown"], []])
def test_fetch_result_rejects_foreign_jobs(job_ids: list[str]) -> None:
    connection = LocalConnection()
    remote_results = connection.submit(_sequence(n_qubits=3), backend_configuration=_config())

    with pytest.raises(RemoteResultsError, match="does not contain jobs"):
        connection._fetch_result(remote_results.batch_id, job_ids)


def test_open_batch_is_not_supported() -> None:
    connection = LocalConnection()
    check.is_false(connection.supports_open_batch())

    with pytest.raises(NotImplementedError, match="open batches"):
        connection.submit(_sequence(n_qubits=3), open=True, backend_configuration=_config())


def test_too_many_qubits_is_not_supported() -> None:
    with pytest.raises(NotImplementedError, match=f"limit is {_QUBIT_LIMIT}"):
        LocalConnection().submit(
            _sequence(n_qubits=_QUBIT_LIMIT), backend_configuration=_config()
        )


@pytest.mark.parametrize("num_shots", [1, 20])
def test_runs_through_remote_emulator(num_shots: int) -> None:
    """The connection drives a `RemoteEmulator` without any credentials."""
    emulator = RemoteEmulator(connection=LocalConnection(), num_shots=num_shots)
    results = emulator.run(_program(n_qubits=3)).results()

    counts = results.get_result(results.get_result_tags()[0], 1.0)
    check.equal(sum(counts.values()), num_shots)

def test_end_to_end() -> None:

    Q = matrix.tensor([
        [-0.2, 0.0, 1.0],
        [ 0.0, 0.0, 1.5],
        [ 1.0, 1.5, 0.0],
    ])
    instance = Instance(Q)
    connection = LocalConnection()
    device = qoolqit.AnalogDevice()
    backend = RemoteEmulator(connection=connection)

    register = embedding.blade.embed(instance)
    drive = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device)
    job = solving.analog_quantum_sampling.solve(register, drive, backend=backend, device=device)
    check.equal(job.get_status(), JobStatus.DONE)

    solution = Solution.from_results(job.results(), instance)
    print("\nSolution:")
    print(analysis.to_dataframe([solution]))

    reloaded_job = retrieve_remote_job(connection, job.job_id(), batch_id=get_batch_id(job))
    check.equal(reloaded_job.get_status(), JobStatus.DONE)

    reloaded_solution = Solution.from_results(reloaded_job.results(), instance)
    print("\nReloaded Solution:")
    print(analysis.to_dataframe([reloaded_solution]))

    torch.testing.assert_close(reloaded_solution.bitstrings, solution.bitstrings)
    torch.testing.assert_close(reloaded_solution.costs, solution.costs)
    torch.testing.assert_close(reloaded_solution.counts, solution.counts)
    torch.testing.assert_close(reloaded_solution.probabilities, solution.probabilities)
