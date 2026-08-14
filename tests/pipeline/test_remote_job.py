from __future__ import annotations

import io
import pytest
import pytest_check as check
import numpy as np
import random
import torch


from pulser.backend.remote import RemoteConnection
from pulser.backend.results import Results

from qubosolver import (
    Instance,
    Solution,
    Analyzer,
    EmbedderType,
    DriveType,
    matrix,
    LocalEmulator,
    RemoteEmulator,
    embedding,
    drive_shaping,
    solvers,
    transforms,
)
import qubosolver._io.utils as io_utils
from qubosolver.types import protocols

from qoolqit import DigitalAnalogDevice
from qoolqit.execution import (
    retrieve_remote_job,
    get_batch_id,
    job,
    JobStatus,
)
from mock.connection import MockConnection


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("embedding_method", list(EmbedderType))
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no_pre"])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_quantum_remote_job(
    make_mock_connection: type[MockConnection],
    drive_method: str,
    embedding_method: str,
    preprocessing: bool,
    dmm: bool,
) -> None:
    if drive_method == DriveType.BAYESIAN_SEARCH:
        pytest.skip(reason="Does not work with the Bayesian-search drive shaping method")

    seed = 7979
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    Q = matrix.tensor(
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
    ) -> tuple[job.Job[Results], Instance]:
        instance = Instance(Q)
        device = DigitalAnalogDevice()

        if preprocessing:
            instance = transforms.variable_fixing.apply_recursively(instance)

        if embedding_method == EmbedderType.BLADE:
            register = embedding.blade.embed(instance)
        else:
            config = embedding.greedy_layout.Config(traps=100)
            register = embedding.greedy_layout.embed(instance, device, config=config)

        num_shots = 50
        backend: protocols.Backend
        if connection is None:
            backend = LocalEmulator(num_shots=num_shots)
        else:
            backend = RemoteEmulator(connection=connection, num_shots=num_shots)

        if drive_method == DriveType.PROPORTIONAL_DIAGONAL:
            drive = drive_shaping.proportional_diagonal.build_drive(
                instance, register, device=device, dmm=dmm
            )
        else:
            drive, _ = drive_shaping.bayesian_search.build_drive(
                instance, register, backend, device, dmm=dmm
            )

        job = solvers.analog_quantum_sampling(register, drive, backend, device)

        return job, instance

    def post(job: job.Job[Results], instance: Instance) -> Solution:
        solution = Solution.from_results(job.results())

        # Post-process fixations of the preprocessing and restore the original QUBO
        if preprocessing:
            assert isinstance(instance, transforms.variable_fixing.Instance)
            solution = transforms.variable_fixing.lift(solution, instance)
            instance = instance._parent_instance
        solution = solvers.iterative_bitflip_local_search(instance, solution)

        solution._compute_costs(instance.matrix)._sort_by_cost()._compute_probabilities()

        return solution

    local_job, local_solver = pre()
    local_solution = post(local_job, local_solver)

    connection = make_mock_connection(local_job.results())
    remote_job, remote_instance = pre(connection)
    assert isinstance(remote_job.results(), Results)

    mock_file = io.BytesIO()
    remote_instance.save(mock_file, remote_instance)
    io_utils.save_string(mock_file, remote_job.job_id())
    io_utils.save_string(mock_file, get_batch_id(remote_job))

    with pytest.raises(ValueError):
        invalid_job = retrieve_remote_job(connection, "invalid_job_id", batch_id="invalid_batch_id")
        invalid_job.get_status()

    mock_file.seek(0)
    remote_instance_2 = remote_instance.load(mock_file)
    job_id_2 = io_utils.load_string(mock_file)
    batch_id_2 = io_utils.load_string(mock_file)
    remote_job_2 = retrieve_remote_job(connection, job_id_2, batch_id=batch_id_2)
    check.equal(remote_job_2.get_status(), JobStatus.DONE)
    remote_solution = post(remote_job_2, remote_instance_2)

    torch.testing.assert_close(remote_solution.bitstrings, local_solution.bitstrings)
    torch.testing.assert_close(remote_solution.costs, local_solution.costs)
    torch.testing.assert_close(remote_solution.probabilities, local_solution.probabilities)
    torch.testing.assert_close(remote_solution.counts, local_solution.counts)

    analyzer = Analyzer([local_solution, remote_solution], labels=["local", "remote"])
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
