from __future__ import annotations

import pytest
import pytest_check as check
import torch

from qubosolver import torch_rng


@pytest.mark.flaky(reruns=4)
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


@pytest.mark.usefixtures("restore_rng_state")
def test_global_rngs_follow_global_seed() -> None:
    def draw() -> torch.Tensor:
        return torch.randint(0, 2, (64,), generator=torch_rng("global"))

    torch.manual_seed(1)
    c1, c2 = draw(), draw()
    torch.manual_seed(1)
    c3, c4 = draw(), draw()

    torch.testing.assert_close(c1, c3)
    torch.testing.assert_close(c2, c4)
    check.is_false(torch.equal(c1, c2))


@pytest.mark.usefixtures("restore_rng_state")
def test_global_rng_advances_global_state() -> None:
    torch.manual_seed(1)
    before = torch.get_rng_state()
    torch_rng("global")
    check.is_false(torch.equal(before, torch.get_rng_state()))


@pytest.mark.usefixtures("restore_rng_state")
def test_unseeded_rng_ignores_global_seed() -> None:
    torch.manual_seed(1)
    before = torch.get_rng_state()
    a = torch.randint(0, 2, (64,), generator=torch_rng())
    torch.manual_seed(1)
    b = torch.randint(0, 2, (64,), generator=torch_rng())

    check.is_false(torch.equal(a, b))
    torch.testing.assert_close(torch.get_rng_state(), before)
