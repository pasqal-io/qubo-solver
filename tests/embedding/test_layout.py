# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import math

import numpy as np
import pytest
import pytest_check as check
import scipy
from pulser.register.special_layouts import SquareLatticeLayout

from qubosolver import embedding
from qubosolver.embedding._algorithms.greedy.layout import get_layout


@pytest.fixture(
    params=[embedding.Lattice.TRIANGULAR, embedding.Lattice.SQUARE], ids=["triangular", "square"]
)
def layout_type(request: pytest.FixtureRequest) -> embedding.Lattice:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[2, 3, 5, 10, 37, 100], ids=str)
def n_traps(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


def test_get_layout_returns_tensor_shape(layout_type: embedding.Lattice, n_traps: int) -> None:
    coords = get_layout(layout_type=layout_type, n_traps=n_traps)
    assert isinstance(coords, np.ndarray)  # noqa: TID251 (runtime check, not a hint)
    assert coords.ndim == 2
    assert coords.shape == (n_traps, 2)
    assert np.unique(coords, axis=0).shape[0] == coords.shape[0]

    pdists = scipy.spatial.distance.pdist(coords)
    assert min(pdists) == pytest.approx(1.0)

    assert np.linalg.norm(coords, axis=1).max().item() <= 1 / math.sqrt(2) * math.ceil(
        math.sqrt(n_traps)
    )


@pytest.mark.parametrize("layout_str", ["SQUARE", "TRIANGULAR"])
def test_get_layout_accepts_case_insensitive_strings(layout_str: str) -> None:
    coords1 = get_layout(layout_type=layout_str, n_traps=10)
    coords2 = get_layout(layout_type=layout_str.lower(), n_traps=10)
    assert np.array_equal(coords1, coords2)


@pytest.mark.parametrize("n_traps", [2, 5, 10, 50])
def test_get_layout_square_spacing_is_one(n_traps: int) -> None:
    coords = get_layout(layout_type=embedding.Lattice.SQUARE, n_traps=n_traps)
    assert np.allclose(coords, coords.round())
    d = scipy.spatial.distance.cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    assert d.min().item() == pytest.approx(1.0)


@pytest.mark.parametrize("n_traps", [1, 2, 3, 5, 10, 37, 100])
def test_get_layout_square_is_compact(n_traps: int) -> None:
    coords = get_layout(layout_type=embedding.Lattice.SQUARE, n_traps=n_traps)
    n = math.ceil(math.sqrt(2 * n_traps))
    candidates = np.array(SquareLatticeLayout(n, n, spacing=1).coords, dtype=coords.dtype)
    cand_set = {tuple(p.tolist()) for p in candidates}
    sel_set = {tuple(p.tolist()) for p in coords}
    assert sel_set.issubset(cand_set)
    cand_d2 = (candidates**2).sum(axis=1)
    sel_d2 = np.square(coords).sum(axis=1)
    kth = np.sort(cand_d2)[n_traps - 1].item()
    assert sel_d2.max().item() == pytest.approx(kth)
    closer = candidates[cand_d2 < kth]
    closer_set = {tuple(p.tolist()) for p in closer}
    assert closer_set.issubset(sel_set)


def test_empty_layout(layout_type: embedding.Lattice) -> None:
    coords = get_layout(layout_type=layout_type, n_traps=0)
    check.equal(coords.shape, (0, 2))


def test_get_layout_rejects_unsupported_layout() -> None:
    with pytest.raises(ValueError, match="Unsupported layout_type: 'hexagonal'"):
        get_layout(layout_type="hexagonal", n_traps=3)
