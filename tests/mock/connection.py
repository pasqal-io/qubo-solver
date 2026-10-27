# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

from pulser.backend.remote import (
    BatchStatus,
    JobStatus,
    RemoteConnection,
    RemoteResults,
)
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pulser import Sequence as PulserSequence
    from pulser.backend.results import Results


class MockConnection(RemoteConnection):
    def __init__(self, result: Results, running_iterations: int = 0) -> None:
        self.result: Results = result
        self.results: dict[str, Results] = {}

        self._status_calls = 0
        self._support_open_batch = True
        self._got_closed = ""
        self._progress_calls = 0
        self._running_iterations = running_iterations

    @override
    def submit(
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

    @override
    def _fetch_result(self, batch_id: str, job_ids: list[str] | None = None) -> tuple[Results, ...]:
        return (self.results[batch_id],)

    def _query_job_progress(self, batch_id: str) -> Mapping[str, tuple[JobStatus, Results | None]]:
        if self._running_iterations > 0:
            self._running_iterations -= 1
            return {"efgh": (JobStatus.RUNNING, None)}
        if batch_id not in self.results:
            return {"efgh": (JobStatus.ERROR, None)}
        return {"efgh": (JobStatus.DONE, self.results[batch_id])}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        # Allow special batch-ids for testing
        for status in BatchStatus:
            if status.name in batch_id:
                return status
        if batch_id not in self.results:
            return BatchStatus.ERROR
        return BatchStatus.DONE

    @override
    def _get_job_ids(self, batch_id: str) -> list[str]:
        return ["efgh"]

    def _close_batch(self, batch_id: str) -> None:
        self._got_closed = batch_id

    def supports_open_batch(self) -> bool:
        return bool(self._support_open_batch)
