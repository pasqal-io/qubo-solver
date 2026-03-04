from __future__ import annotations


from pulser_pasqal import PasqalCloud
from pulser.backend.remote import (
    BatchStatus,
    JobStatus,
    RemoteResults,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pulser import Sequence as PulserSequence
    from pulser.backend.results import Results
    from typing import Any, Dict, Mapping


class MockConnection(PasqalCloud):
    def __init__(self, result: Results) -> None:
        self.result: Results = result
        self.result.bitstring_counts = self.result.final_bitstrings  # type: ignore[attr-defined]
        self.results: Dict[str, Results] = dict()

        self._status_calls = 0
        self._support_open_batch = True
        self._got_closed = ""
        self._progress_calls = 0

    def submit(  # type: ignore[override]
        self,
        sequence: PulserSequence,
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

    def _query_job_progress(self, batch_id: str) -> Mapping[str, tuple[JobStatus, Results | None]]:
        if batch_id not in self.results.keys():
            return {batch_id: (JobStatus.ERROR, None)}
        return {batch_id: (JobStatus.DONE, self.results[batch_id])}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        # Allow special batch-ids for testing
        for status in BatchStatus:
            if status.name in batch_id:
                return status
        if batch_id not in self.results.keys():
            return BatchStatus.ERROR
        return BatchStatus.DONE

    def _close_batch(self, batch_id: str) -> None:
        self._got_closed = batch_id

    def supports_open_batch(self) -> bool:
        return bool(self._support_open_batch)
