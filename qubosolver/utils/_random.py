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

try:
    import torch  # deptry: ignore[DEP004] (optional, installed via pulser[torch])
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Generator


def manual_seed(seed: int) -> np.random.Generator:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    return numpy_rng(seed)


def reset_seed() -> np.random.Generator:
    """Reset random seeds to a non-deterministic state."""
    np.random.seed(None)
    random.seed(None)
    if torch is not None:
        torch.seed()
    return numpy_rng()


@contextmanager
def seed_context(seed: int) -> Generator[np.random.Generator]:
    """Temporarily seed numpy/random/torch, restoring prior state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state() if torch is not None else None
    try:
        yield manual_seed(seed)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)


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
