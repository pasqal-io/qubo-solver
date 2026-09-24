from __future__ import annotations

import io
import random
from typing import get_args

import numpy as np
import pytest
import pytest_check as check
import torch
from mock.connection import MockConnection
from pulser.backend.remote import RemoteConnection
from pulser.backend.results import Results
from qoolqit import AnalogDeviceWithDMM
from qoolqit.execution import (
    JobStatus,
    get_batch_id,
    job,
    retrieve_remote_job,
)

import qubosolver._io.utils as io_utils
from qubosolver import (
    Instance,
    LocalEmulator,
    RemoteEmulator,
    Solution,
    drive_shaping,
    embedding,
    matrix,
    solving,
    transforms,
)
from qubosolver.solver.config.drive_shaping import _DriveShapingAlgorithm
from qubosolver.solver.config.embedding import _EmbeddingAlgorithm
from qubosolver.types import protocols
from qubosolver.utils import analysis


def _pre_remote_job(
    Q: matrix.Matrix,
    *,
    drive_method: _DriveShapingAlgorithm,
    embedding_method: _EmbeddingAlgorithm,
    preprocessing: bool,
    dmm: bool,
    connection: RemoteConnection | None = None,
) -> tuple[job.Job[Results], Instance]:
    instance = Instance(Q)
    device = AnalogDeviceWithDMM()

    if preprocessing:
        instance = transforms.variable_fixing.apply_recursively(instance)

    if embedding_method == "blade":
        register = embedding.blade.embed(instance)
    else:
        config = embedding.greedy_layout.Config(traps=100)
        register = embedding.greedy_layout.embed(instance, config=config)

    num_shots = 50
    backend: protocols.Backend
    if connection is None:
        backend = LocalEmulator(num_shots=num_shots)
    else:
        backend = RemoteEmulator(connection=connection, num_shots=num_shots)

    if drive_method == "proportional_diagonal":
        drive = drive_shaping.proportional_diagonal.build_drive(
            instance, register, device=device, dmm=dmm
        )
    else:
        _, drive = solving.drive_bayesian_search.solve(
            instance, register, backend=backend, device=device, dmm=dmm
        )

    program = solving.analog_quantum_sampling.compile(register, drive, device)
    job_ = backend.run(program)

    return job_, instance


def _post_remote_job(
    job_: job.Job[Results], instance: Instance, *, preprocessing: bool
) -> Solution:
    solution = Solution.from_results(job_.results(), instance)

    # Post-process fixations of the preprocessing and restore the original QUBO
    if preprocessing:
        assert isinstance(instance, transforms.variable_fixing.Instance)
        solution = transforms.variable_fixing.lift(solution, instance)
        instance = instance._parent_instance
    solution = solving.iterative_bitflip_local_search.solve(instance, starts=solution)

    solution._compute_costs(instance.matrix)._sort_by_cost()._compute_probabilities()

    return solution


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("drive_method", get_args(_DriveShapingAlgorithm))
@pytest.mark.parametrize("embedding_method", get_args(_EmbeddingAlgorithm))
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no_pre"])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_quantum_remote_job(
    make_mock_connection: type[MockConnection],
    drive_method: _DriveShapingAlgorithm,
    embedding_method: _EmbeddingAlgorithm,
    preprocessing: bool,
    dmm: bool,
) -> None:
    if drive_method == "bayesian_search":
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

    local_job, local_instance = _pre_remote_job(
        Q,
        drive_method=drive_method,
        embedding_method=embedding_method,
        preprocessing=preprocessing,
        dmm=dmm,
    )
    local_solution = _post_remote_job(local_job, local_instance, preprocessing=preprocessing)

    connection = make_mock_connection(local_job.results())
    remote_job, remote_instance = _pre_remote_job(
        Q,
        drive_method=drive_method,
        embedding_method=embedding_method,
        preprocessing=preprocessing,
        dmm=dmm,
        connection=connection,
    )
    assert isinstance(remote_job.results(), Results)

    mock_file = io.BytesIO()
    remote_instance.save(mock_file)
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
    remote_solution = _post_remote_job(remote_job_2, remote_instance_2, preprocessing=preprocessing)

    torch.testing.assert_close(remote_solution.bitstrings, local_solution.bitstrings)
    torch.testing.assert_close(remote_solution.costs, local_solution.costs)
    torch.testing.assert_close(remote_solution.probabilities, local_solution.probabilities)
    torch.testing.assert_close(remote_solution.counts, local_solution.counts)

    df_all = analysis.to_dataframe([local_solution, remote_solution], labels=["local", "remote"])
    print(f"\n{df_all}")

    expected_solutions = ["00111", "01011"]

    for solution in (local_solution, remote_solution):
        check.is_true(solution.check_consistency(instance=Instance(Q), throw=True))
        N = min(len(solution), 2)
        for i in range(N):
            s = solution[i]
            check.is_in(s.string, expected_solutions)
            check.almost_equal(s.cost, -27.288260)
        cumulative_probability = sum(solution[i].probability for i in range(N))
        check.greater(cumulative_probability, 0.9)
