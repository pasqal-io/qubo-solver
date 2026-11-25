# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import random
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check
from IPython.core import getipython
from IPython.core.interactiveshell import InteractiveShell

from qubosolver import numpy_rng
from qubosolver.utils._random import manual_seed, register_seeded_magic, reset_seed, seed_context

if TYPE_CHECKING:
    from collections.abc import Generator


def draw() -> tuple[float, float, np.ndarray]:
    return random.random(), np.random.rand(), numpy_rng("global").random(4)  # noqa: S311 (seeded test data, not cryptographic)


@pytest.fixture
def shell() -> Generator[InteractiveShell]:
    yield InteractiveShell.instance()
    InteractiveShell.clear_instance()


@pytest.mark.usefixtures("restore_rng_state")
def test_reset_seed_returns_generator() -> None:
    check.is_instance(reset_seed(), np.random.Generator)


@pytest.mark.usefixtures("restore_rng_state")
def test_seed_context_is_reproducible_and_restores_state() -> None:
    manual_seed(1)
    expected_after = draw()
    manual_seed(1)

    with seed_context(2) as rng:
        check.is_instance(rng, np.random.Generator)
        first = draw()
    with seed_context(2):
        second = draw()
    after = draw()

    check.equal(first[:2], second[:2])
    np.testing.assert_allclose(first[2], second[2], strict=True)
    check.equal(after[:2], expected_after[:2])
    np.testing.assert_allclose(after[2], expected_after[2], strict=True)


@pytest.mark.usefixtures("restore_rng_state")
def test_seeded_magic_seeds_the_cell(shell: InteractiveShell) -> None:
    register_seeded_magic()
    shell.run_cell("import random", store_history=False)
    shell.run_cell_magic("seeded", "3", "x = random.random()")
    random.seed(3)
    check.equal(shell.user_ns["x"], random.random())  # noqa: S311 (seeded test data, not cryptographic)


def test_seeded_magic_requires_a_shell(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first call registers the magic, the second runs inside it.
    shells = iter([shell, None])
    monkeypatch.setattr(getipython, "get_ipython", lambda: next(shells))
    register_seeded_magic()
    with pytest.raises(RuntimeError, match="requires a running IPython shell"):
        shell.run_cell_magic("seeded", "3", "pass")


def test_register_seeded_magic_requires_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "IPython.core.getipython", None)
    with pytest.raises(ImportError, match="requires IPython"):
        register_seeded_magic()
