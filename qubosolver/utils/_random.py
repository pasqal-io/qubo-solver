# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from qubosolver.types import numpy_rng

if TYPE_CHECKING:
    from collections.abc import Generator


def manual_seed(seed: int) -> np.random.Generator:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    return numpy_rng(seed)


def reset_seed() -> np.random.Generator:
    """Reset random seeds to a non-deterministic state."""
    np.random.seed(None)
    random.seed(None)
    return numpy_rng()


@contextmanager
def seed_context(seed: int) -> Generator[np.random.Generator]:
    """Temporarily seed numpy/random, restoring prior state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield manual_seed(seed)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def register_seeded_magic() -> None:
    """Register the ``%%seeded <seed>`` cell magic in the running IPython kernel."""
    try:
        from IPython.core.getipython import (  # noqa: PLC0415 (optional dependency)  # deptry: ignore[DEP004]
            get_ipython,
        )
        from IPython.core.magic import (  # noqa: PLC0415 (optional dependency)  # deptry: ignore[DEP004]
            register_cell_magic,
        )
    except ImportError as e:
        msg = (
            "register_seeded_magic() requires IPython. "
            "Install it with: pip install 'qubo-solver[dev]'"
        )
        raise ImportError(msg) from e

    @register_cell_magic
    def seeded(line: str, cell: str) -> None:
        seed = int(line.strip())
        shell = get_ipython()
        if shell is None:
            msg = "%%seeded requires a running IPython shell."
            raise RuntimeError(msg)
        with seed_context(seed):
            shell.run_cell(cell)
