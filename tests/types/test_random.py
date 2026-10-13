from __future__ import annotations

import pytest
import pytest_check as check
import torch

from qubosolver import torch_rng


@pytest.mark.flaky(max_runs=5)
def test_unseeded_rngs_differ() -> None:
    # A fresh torch.Generator always starts from the same fixed seed, so unseeded
    # generators must be explicitly randomized to produce different streams.
    a = torch.randint(0, 2, (64,), generator=torch_rng())
    b = torch.randint(0, 2, (64,), generator=torch_rng())
    check.is_false(torch.equal(a, b))


def test_seeded_rngs_match() -> None:
    a = torch.randint(0, 2, (64,), generator=torch_rng(0))
    b = torch.randint(0, 2, (64,), generator=torch_rng(0))
    check.is_true(torch.equal(a, b))
