"""A remote connection that runs sequences locally, for tutorials and tests.

It exposes the same interface as a real cloud connection (batches, jobs,
statuses, lazily fetched results) while emulating everything locally, so no
credentials are needed.
"""

from __future__ import annotations

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
    """A RemoteConnection that emulates sequences with QuTiP.

    Sequences are executed synchronously during `submit`, as a single job per
    batch, so a batch is already done by the time `submit` returns. Results are
    kept in memory. Open batches and large registers are not supported.

    Example:
        ```python
        emulator = RemoteEmulator(connection=LocalConnection(), num_shots=1000)
        ```
    """

    def __init__(self) -> None:
        """Instantiate a connection with no submitted batch."""
        self._batches: dict[str, Results] = {}

    def submit(
        self,
        sequence: Sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs: Any,
    ) -> RemoteResults:
        """Emulate a sequence and store its results as a one-job batch.

        `wait` and `batch_id` are ignored, as is every keyword argument other
        than `backend_configuration`, the emulation config carrying the
        observables to compute.
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
        self._batches[batch_id] = QutipBackendV2(
            sequence, config=kwargs.get("backend_configuration")
        ).run()
        return RemoteResults(batch_id, self)

    def supports_open_batch(self) -> bool:
        """Return whether open batches are supported, which they are not."""
        return False

    def _fetch_result(self, batch_id: str, job_ids: list[str] | None) -> TypingSequence[Results]:
        """Return the results of a batch, whose only job `job_ids` must match."""
        results = self._batch(batch_id)
        if job_ids is not None and job_ids != self._get_job_ids(batch_id):
            raise RemoteResultsError(f"Batch {batch_id!r} does not contain jobs {job_ids}.")
        return (results,)

    def _query_job_progress(self, batch_id: str) -> Mapping[str, tuple[JobStatus, Results | None]]:
        """Return the status and results of the batch's single job, always done."""
        return {self._job_id(batch_id): (JobStatus.DONE, self._batch(batch_id))}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Return whether the batch is known to this connection."""
        return BatchStatus.DONE if batch_id in self._batches else BatchStatus.ERROR

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Return the ID of the batch's single job, rejecting an unknown batch."""
        # Fabricating an ID for an unknown batch would let it pass the
        # validation `RemoteResults` performs on the job IDs it is given.
        self._batch(batch_id)
        return [self._job_id(batch_id)]

    def _batch(self, batch_id: str) -> Results:
        """Return the results stored for a batch, or raise if it is unknown."""
        if batch_id not in self._batches:
            raise RemoteResultsError(f"Unknown batch {batch_id!r} for this connection.")
        return self._batches[batch_id]

    @staticmethod
    def _job_id(batch_id: str) -> str:
        """Return the ID of the single job belonging to a batch."""
        return f"{batch_id}-0"
