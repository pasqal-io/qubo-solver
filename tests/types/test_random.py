# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check

from qubosolver import numpy_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.flaky(reruns=4)
def test_unseeded_rngs_differ() -> None:
    # Unseeded generators draw their seed from a non-deterministic source.
    a = numpy_rng().integers(0, 2, (64,))
    b = numpy_rng().integers(0, 2, (64,))
    check.is_false(np.array_equal(a, b))


def test_seeded_rngs_match() -> None:
    a = numpy_rng(0).integers(0, 2, (64,))
    b = numpy_rng(0).integers(0, 2, (64,))
    check.is_true(np.array_equal(a, b))


@pytest.mark.usefixtures("restore_rng_state")
def test_global_rngs_follow_global_seed() -> None:
    def draw() -> NDArray[np.int64]:
        return numpy_rng("global").integers(0, 2, (64,))

    np.random.seed(1)
    c1, c2 = draw(), draw()
    np.random.seed(1)
    c3, c4 = draw(), draw()

    np.testing.assert_array_equal(c1, c3)
    np.testing.assert_array_equal(c2, c4)
    check.is_false(np.array_equal(c1, c2))


@pytest.mark.usefixtures("restore_rng_state")
def test_global_rng_advances_global_state() -> None:
    np.random.seed(1)
    before = np.random.get_state(legacy=False)
    numpy_rng("global")
    after = np.random.get_state(legacy=False)
    check.not_equal(before["state"]["pos"], after["state"]["pos"])


@pytest.mark.usefixtures("restore_rng_state")
def test_unseeded_rng_ignores_global_seed() -> None:
    np.random.seed(1)
    before = np.random.get_state(legacy=False)
    a = numpy_rng().integers(0, 2, (64,))
    np.random.seed(1)
    b = numpy_rng().integers(0, 2, (64,))

    check.is_false(np.array_equal(a, b))
    after = np.random.get_state(legacy=False)
    np.testing.assert_array_equal(after["state"]["key"], before["state"]["key"])
    check.equal(after["state"]["pos"], before["state"]["pos"])
