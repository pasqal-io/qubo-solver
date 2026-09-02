"""A remote connection that executes sequences locally.

Intended for tutorials and tests: it exposes the same interface as a real cloud
connection (batches, jobs, statuses, lazily fetched results) while running
everything on a local emulator, so no credentials are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence as TypingSequence
from uuid import uuid4

from pulser.backend.remote import (
    BatchStatus,
    JobStatus,
    RemoteConnection,
    RemoteResults,
    RemoteResultsError,
)
from pulser.backend.results import Results
from pulser.sequence import Sequence
from pulser_simulation import QutipBackendV2

from qubosolver.types.backends import _SV_THRESHOLD

# From this register size on, QuTiP emulation becomes intractable.
_QUBIT_LIMIT = _SV_THRESHOLD


class LocalConnection(RemoteConnection):
    """A [`RemoteConnection`][pulser.backend.remote.RemoteConnection] backed by local emulation.

    Sequences are executed synchronously during [`submit`][] on a local QuTiP
    emulator, as a single job per batch. A batch is therefore already `DONE` by
    the time `submit` returns. Open batches are not supported, and QuTiP
    restricts this connection to small registers.

    Results are kept in memory. When `storage_dir` is given, each batch is also
    written to disk as JSON so it can be inspected after the fact.

    Args:
        storage_dir: Optional directory in which to save job data. When omitted,
            results are only kept in memory for the lifetime of the connection.

    Example:
        ```python
        from qubosolver import RemoteEmulator
        from qubosolver.utils._local_connection import LocalConnection

        emulator = RemoteEmulator(connection=LocalConnection(), num_shots=1000)
        ```
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._batches: dict[str, Results] = {}
        self._storage_dir = Path(storage_dir) if storage_dir is not None else None
        if self._storage_dir is not None:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        sequence: Sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        *,
        backend_configuration: Any = None,
        **kwargs: Any,
    ) -> RemoteResults:
        """Execute the sequence locally and store the results as a one-job batch.

        Args:
            sequence: The sequence to execute.
            open: Whether to open a batch, which this connection cannot do.
            backend_configuration: The emulation config to run with, forwarded
                by the calling backend. Carries the observables to compute.
            wait: Ignored, along with `batch_id` and extra `kwargs` such as
                `job_params` and `device_type`.
            batch_id: Ignored; every submission creates its own batch.
            **kwargs: Ignored.

        Returns:
            The results of the newly submitted batch.

        Raises:
            NotImplementedError: If an open batch is requested, or if the
                register is too large to emulate locally.
        """
        if open:
            raise NotImplementedError(
                f"{type(self).__name__} does not support open batches. "
                "Submit each batch on its own instead."
            )

        sequence = self._add_measurement_to_sequence(sequence)
        n_qubits = len(sequence.register.qubit_ids)
        if n_qubits >= _QUBIT_LIMIT:
            raise NotImplementedError(
                f"{type(self).__name__} emulates sequences with QuTiP, which is intractable "
                f"for the {n_qubits} qubits of this sequence (limit is {_QUBIT_LIMIT}). "
                "Use a real connection, or a local emulator backend suited to this size."
            )

        batch_id = str(uuid4())
        results = QutipBackendV2(sequence, config=backend_configuration).run()
        self._batches[batch_id] = results

        if self._storage_dir is not None:
            job_data = {self._job_id(batch_id): json.loads(results.to_abstract_repr())}
            (self._storage_dir / f"{batch_id}.json").write_text(json.dumps(job_data))

        return RemoteResults(batch_id, self)

    def _fetch_result(self, batch_id: str, job_ids: list[str] | None) -> TypingSequence[Results]:
        """Return the results of a batch.

        Args:
            batch_id: The batch to fetch the results of.
            job_ids: When given, must be the batch's own single job.

        Returns:
            The batch's single results.

        Raises:
            RemoteResultsError: If the batch is unknown, or if `job_ids` does
                not match the batch's job.
        """
        results = self._batch(batch_id)
        if job_ids is not None and job_ids != self._get_job_ids(batch_id):
            raise RemoteResultsError(f"Batch {batch_id!r} does not contain jobs {job_ids}.")
        return (results,)

    def _query_job_progress(self, batch_id: str) -> Mapping[str, tuple[JobStatus, Results | None]]:
        """Return the status and results of the batch's job.

        Args:
            batch_id: The batch to query.

        Returns:
            A mapping of job ID to its status and results. Since execution is
            synchronous, the job is `DONE` and carries its results.

        Raises:
            RemoteResultsError: If the batch is unknown.
        """
        return {self._job_id(batch_id): (JobStatus.DONE, self._batch(batch_id))}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Return the status of a batch, always `DONE` once submitted.

        Args:
            batch_id: The batch to get the status of.

        Returns:
            `BatchStatus.DONE`.
        """
        return BatchStatus.DONE if batch_id in self._batches else BatchStatus.ERROR

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Return the ID of the batch's single job.

        Args:
            batch_id: The batch to get the job IDs of.

        Returns:
            A single job ID.

        Raises:
            RemoteResultsError: If the batch is unknown. Fabricating an ID here
                would let an unknown batch pass `RemoteResults`' validation.
        """
        self._batch(batch_id)
        return [self._job_id(batch_id)]

    def supports_open_batch(self) -> bool:
        """Whether this connection supports open batches.

        Returns:
            `False`; each submission is executed and closed immediately.
        """
        return False

    def _batch(self, batch_id: str) -> Results:
        """Return the stored results of a batch.

        Args:
            batch_id: The batch to look up.

        Returns:
            The batch's results.

        Raises:
            RemoteResultsError: If this connection has no such batch.
        """
        if batch_id not in self._batches:
            raise RemoteResultsError(f"Unknown batch {batch_id!r} for this connection.")
        return self._batches[batch_id]

    @staticmethod
    def _job_id(batch_id: str) -> str:
        """Return the ID of the single job belonging to a batch.

        Args:
            batch_id: The batch the job belongs to.

        Returns:
            The job ID.
        """
        return f"{batch_id}-0"
