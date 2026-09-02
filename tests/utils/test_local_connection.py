from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import pulser
import pytest
import pytest_check as check
import qoolqit
import torch
from pulser.backend import EmulationConfig
from pulser.backend.default_observables import BitStrings
from pulser.backend.remote import BatchStatus, JobStatus, RemoteResults, RemoteResultsError
from pulser.backend.results import Results

from qubosolver import Instance, RemoteEmulator, drive_shaping, matrix, vector
from qubosolver.utils._local_connection import _QUBIT_LIMIT, LocalConnection

NUM_SHOTS = 50


def _program(n_qubits: int) -> qoolqit.QuantumProgram:
    """Build a compiled program on a compact square grid of `n_qubits` atoms."""
    side = math.ceil(math.sqrt(n_qubits))
    coordinates = matrix.tensor(
        [[float(i % side), float(i // side)] for i in range(n_qubits)]
    )
    register = qoolqit.Register.from_coordinates(coordinates)
    instance = Instance(
        matrix.as_tensor(register.interaction_matrix())
        + torch.diag(vector.tensor([-1.0] * n_qubits))
    )
    device = qoolqit.AnalogDevice()
    drive = drive_shaping.proportional_diagonal.build_drive(
        instance, register, dmm=False, device=device
    )
    program = qoolqit.QuantumProgram(register, drive)
    program.compile_to(device, profile="max_energy", device_max_duration_ratio=0.999)
    return program


@pytest.fixture(scope="module")
def sequence() -> pulser.Sequence:
    """A compiled 3-qubit sequence, cheap enough to emulate repeatedly."""
    return _program(3).compiled_sequence


@pytest.fixture
def config() -> EmulationConfig:
    """An emulation config collecting bitstrings at the end of the sequence."""
    return EmulationConfig(
        observables=[BitStrings(num_shots=NUM_SHOTS)], default_evaluation_times=[1.0]
    )


@pytest.fixture
def submitted(
    sequence: pulser.Sequence, config: EmulationConfig
) -> tuple[LocalConnection, RemoteResults]:
    """A connection with a single submitted batch."""
    connection = LocalConnection()
    return connection, connection.submit(sequence, backend_configuration=config)


def test_submit_returns_completed_bitstring_results(
    submitted: tuple[LocalConnection, RemoteResults],
) -> None:
    connection, remote_results = submitted

    check.equal(remote_results.get_batch_status(), BatchStatus.DONE)
    check.equal(len(remote_results.results), 1)

    results = remote_results.results[0]
    check.is_instance(results, Results)
    check.equal(results.get_result_tags(), ["bitstrings"])

    counts = results.get_result("bitstrings", 1.0)
    check.equal(sum(counts.values()), NUM_SHOTS)
    check.is_true(all(len(bitstring) == 3 for bitstring in counts))


def test_batch_holds_a_single_job(submitted: tuple[LocalConnection, RemoteResults]) -> None:
    connection, remote_results = submitted
    batch_id = remote_results.batch_id

    check.equal(remote_results.job_ids, [f"{batch_id}-0"])
    check.equal(connection._get_job_ids(batch_id), [f"{batch_id}-0"])


def test_query_job_progress_reports_done_with_results(
    submitted: tuple[LocalConnection, RemoteResults],
) -> None:
    connection, remote_results = submitted
    progress = connection._query_job_progress(remote_results.batch_id)

    check.equal(list(progress), remote_results.job_ids)
    status, results = progress[remote_results.job_ids[0]]
    check.equal(status, JobStatus.DONE)
    check.is_instance(results, Results)
    check.equal(list(remote_results.get_available_results()), remote_results.job_ids)


def test_each_submission_creates_its_own_batch(
    sequence: pulser.Sequence, config: EmulationConfig
) -> None:
    connection = LocalConnection()

    first = connection.submit(sequence, backend_configuration=config)
    second = connection.submit(sequence, batch_id=first.batch_id, backend_configuration=config)

    # `batch_id` is ignored, so the second submission does not extend the first.
    check.not_equal(first.batch_id, second.batch_id)
    check.equal(len(connection._batches), 2)
    check.equal(len(second.results), 1)


def test_unknown_batch_reports_error_status() -> None:
    """A status query answers with a status rather than raising."""
    check.equal(LocalConnection()._get_batch_status("unknown"), BatchStatus.ERROR)


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


def test_fetch_result_accepts_the_batch_own_job(
    submitted: tuple[LocalConnection, RemoteResults],
) -> None:
    connection, remote_results = submitted
    batch_id = remote_results.batch_id

    check.equal(len(connection._fetch_result(batch_id, None)), 1)
    check.equal(len(connection._fetch_result(batch_id, connection._get_job_ids(batch_id))), 1)
    check.equal(len(RemoteResults(batch_id, connection, job_ids=remote_results.job_ids).results), 1)


@pytest.mark.parametrize("job_ids", [["unknown"], []])
def test_fetch_result_rejects_foreign_jobs(
    submitted: tuple[LocalConnection, RemoteResults], job_ids: list[str]
) -> None:
    connection, remote_results = submitted

    with pytest.raises(RemoteResultsError, match="does not contain jobs"):
        connection._fetch_result(remote_results.batch_id, job_ids)


def test_open_batch_is_not_supported(
    sequence: pulser.Sequence, config: EmulationConfig
) -> None:
    connection = LocalConnection()
    check.is_false(connection.supports_open_batch())

    with pytest.raises(NotImplementedError, match="open batches"):
        connection.submit(sequence, open=True, backend_configuration=config)


def test_too_many_qubits_is_not_supported(config: EmulationConfig) -> None:
    sequence = _program(_QUBIT_LIMIT).compiled_sequence

    with pytest.raises(NotImplementedError, match=f"limit is {_QUBIT_LIMIT}"):
        LocalConnection().submit(sequence, backend_configuration=config)


def test_storage_dir_saves_job_results_as_json(
    sequence: pulser.Sequence, config: EmulationConfig, tmp_path: Path
) -> None:
    storage_dir = tmp_path / "jobs"
    connection = LocalConnection(storage_dir=storage_dir)
    remote_results = connection.submit(sequence, backend_configuration=config)

    saved = storage_dir / f"{remote_results.batch_id}.json"
    check.is_true(saved.exists())

    job_data = json.loads(saved.read_text())
    check.equal(list(job_data), remote_results.job_ids)

    restored = Results.from_abstract_repr(json.dumps(job_data[remote_results.job_ids[0]]))
    check.equal(
        restored.get_result("bitstrings", 1.0),
        remote_results.results[0].get_result("bitstrings", 1.0),
    )


def test_without_storage_dir_nothing_is_written(
    sequence: pulser.Sequence, config: EmulationConfig, tmp_path: Path
) -> None:
    LocalConnection().submit(sequence, backend_configuration=config)
    check.equal(list(tmp_path.iterdir()), [])


@pytest.mark.parametrize("num_shots", [1, 20])
def test_runs_through_remote_emulator(num_shots: int) -> None:
    """The connection drives a `RemoteEmulator` without any credentials."""
    emulator = RemoteEmulator(connection=LocalConnection(), num_shots=num_shots)
    results = emulator.run(_program(3)).results()

    counts = results.get_result(results.get_result_tags()[0], 1.0)
    check.equal(sum(counts.values()), num_shots)
