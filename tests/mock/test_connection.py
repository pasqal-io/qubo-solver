# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import pytest_check as check
from pulser.backend.remote import BatchStatus

if TYPE_CHECKING:
    from pulser import Sequence as PulserSequence
    from pulser.backend.results import Results

    from mock.connection import MockConnection


@pytest.fixture
def connection(make_mock_connection: type[MockConnection]) -> MockConnection:
    return make_mock_connection(cast("Results", object()))


@pytest.mark.parametrize("status", list(BatchStatus))
def test_batch_status_from_special_batch_id(
    connection: MockConnection, status: BatchStatus
) -> None:
    check.equal(connection._get_batch_status(f"batch-{status.name}"), status)


def test_batch_status_of_submitted_and_unknown_batches(connection: MockConnection) -> None:
    connection.submit(cast("PulserSequence", None), batch_id="submitted")
    check.equal(connection._get_batch_status("submitted"), BatchStatus.DONE)
    check.equal(connection._get_batch_status("unknown"), BatchStatus.ERROR)


def test_close_batch_records_the_batch_id(connection: MockConnection) -> None:
    connection._close_batch("closed")
    check.equal(connection._got_closed, "closed")


@pytest.mark.parametrize("supported", [True, False])
def test_supports_open_batch(connection: MockConnection, supported: bool) -> None:
    connection._support_open_batch = supported
    check.equal(connection.supports_open_batch(), supported)
