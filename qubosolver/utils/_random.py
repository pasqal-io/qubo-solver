from __future__ import annotations

import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch

from qubosolver.types import torch_rng
from qubosolver.types._checks import no_runtime_typecheck


def manual_seed(seed: int) -> torch.Generator:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


def reset_seed() -> torch.Generator:
    """Reset random seeds to a non-deterministic state."""
    np.random.seed(None)
    torch.seed()
    random.seed(None)
    return torch_rng()


@contextmanager
@no_runtime_typecheck
def seed_context(seed: int) -> Generator[torch.Generator]:
    """Temporarily seed numpy/torch/random, restoring prior state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield manual_seed(seed)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def register_seeded_magic() -> None:
    """Register the ``%%seeded <seed>`` cell magic in the running IPython kernel."""
    from IPython.core.getipython import get_ipython
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def seeded(line: str, cell: str) -> None:
        seed = int(line.strip())
        with seed_context(seed):
            get_ipython().run_cell(cell)
