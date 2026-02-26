from __future__ import annotations

import time
from typing import Protocol, Sequence, TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing import Self

from pulser.backend import Results
from concurrent.futures import ThreadPoolExecutor, TimeoutError, CancelledError

from pulser.backend.remote import (
    RemoteResults,
    RemoteResultsError,
    BatchStatus,
    RemoteConnection,
)

# Implement the concept of concurrent.futures.Future
class BatchConcept(Protocol):

    def cancel(self: Self) -> bool: ...
    def cancelled(self: Self) -> bool: ...
    def running(self: Self) -> bool: ...
    def done(self: Self) -> bool: ...
    def result(self: Self, timeout: float|None = None) -> Sequence[Results]: ...
    def exception(self: Self, timeout: float|None = None) -> BaseException | None: ...

def from_future(function: Callable[..., Sequence[Results]], *args: Any, **kwargs: Any) -> BatchConcept:
    executor = ThreadPoolExecutor()
    future = executor.submit(function, *args, **kwargs)
    executor.shutdown(wait=False)
    return future

def from_remote_results(remote_results: RemoteResults) -> BatchConcept:
    return PulserBatch(remote_results)

def from_batch_id(batch_id: str, connection: RemoteConnection) -> BatchConcept:
    return from_remote_results(RemoteResults(batch_id, connection))

class PulserBatch:

    def __init__(self, remote_results: RemoteResults) -> None:
        self._remote_results = remote_results

    @property
    def batch_id(self) -> str:
        return self._remote_results.batch_id

    def _status(self) -> BatchStatus:
        return self._remote_results.get_batch_status()

    def cancel(self) -> bool:
        raise NotImplementedError("A Pulser Batch cannot be cancelled by user")

    def cancelled(self) -> bool:
        return self._status() == BatchStatus.CANCELED

    def running(self) -> bool:
        return self._status() in [
            BatchStatus.RUNNING,
            BatchStatus.PENDING,
            BatchStatus.PAUSED,
        ]

    def done(self) -> bool:
        return self._status() in [
            BatchStatus.DONE,
            BatchStatus.TIMED_OUT,
            BatchStatus.ERROR,
        ] or self.cancelled()

    def result(self, timeout: float | None = None) -> Sequence[Results]:

        start_time = time.time()

        while True:
            status = self._status()

            # Check if cancelled
            if self.cancelled():
                raise CancelledError("Batch was cancelled")

            # Check if done
            if status == BatchStatus.DONE:
                # Fetch and return the results
                return self._remote_results.results

            # Check if error occurred
            if status == BatchStatus.ERROR:
                # Try to get results which may contain error information
                raise RemoteResultsError("Batch finished with unknown error")

            if status == BatchStatus.TIMED_OUT:
                raise RemoteResultsError("Batch timed out on the remote backend")

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Batch did not complete within {timeout} seconds")

            # Wait a bit before checking again
            time.sleep(0.1)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        try:
            self.result(timeout=timeout)
        except (CancelledError, TimeoutError) as e:
            raise e
        except BaseException as e:
            return e
        return None
