import pytest
import pytest_check as check
import math
import torch
import scipy

from pulser.register.special_layouts import SquareLatticeLayout
from qubosolver import embedding
from qubosolver.embedding._algorithms.greedy.layout import get_layout


@pytest.fixture(params=[embedding.Lattice.TRIANGULAR, embedding.Lattice.SQUARE], ids=["triangular", "square"])
def layout_type(request: pytest.FixtureRequest) -> embedding.Lattice:
    return request.param  # type: ignore


@pytest.fixture(params=[2, 3, 5, 10, 37, 100], ids=str)
def n_traps(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore


def test_get_layout_returns_tensor_shape(layout_type: embedding.Lattice, n_traps: int) -> None:
    coords = get_layout(layout_type=layout_type, n_traps=n_traps)
    assert isinstance(coords, torch.Tensor)
    assert coords.ndim == 2
    assert coords.shape == (n_traps, 2)
    assert coords.unique(dim=0).shape[0] == coords.shape[0]

    pdists = scipy.spatial.distance.pdist(coords)
    assert min(pdists) == pytest.approx(1.0)

    assert torch.linalg.norm(coords, dim=1).max().item() <= 1 / math.sqrt(2) * math.ceil(
        math.sqrt(n_traps)
    )


@pytest.mark.parametrize("layout_str", ["SQUARE", "TRIANGULAR"])
def test_get_layout_accepts_case_insensitive_strings(layout_str: str) -> None:
    coords1 = get_layout(layout_type=layout_str, n_traps=10)
    coords2 = get_layout(layout_type=layout_str.lower(), n_traps=10)
    assert torch.equal(coords1, coords2)


@pytest.mark.parametrize("n_traps", [2, 5, 10, 50])
def test_get_layout_square_spacing_is_one(n_traps: int) -> None:
    coords = get_layout(layout_type=embedding.Lattice.SQUARE, n_traps=n_traps)
    assert torch.allclose(coords, coords.round())
    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float("inf"))
    assert d.min().item() == pytest.approx(1.0)


@pytest.mark.parametrize("n_traps", [1, 2, 3, 5, 10, 37, 100])
def test_get_layout_square_is_compact(n_traps: int) -> None:
    coords = get_layout(layout_type=embedding.Lattice.SQUARE, n_traps=n_traps)
    n = int(math.ceil(math.sqrt(2 * n_traps)))
    candidates = torch.tensor(SquareLatticeLayout(n, n, spacing=1).coords, dtype=coords.dtype)
    cand_set = {tuple(p.tolist()) for p in candidates}
    sel_set = {tuple(p.tolist()) for p in coords}
    assert sel_set.issubset(cand_set)
    cand_d2 = (candidates**2).sum(dim=1)
    sel_d2 = coords.square().sum(dim=1)
    kth = torch.kthvalue(cand_d2, k=n_traps).values.item()
    assert sel_d2.max().item() == pytest.approx(kth)
    closer = candidates[cand_d2 < kth]
    closer_set = {tuple(p.tolist()) for p in closer}
    assert closer_set.issubset(sel_set)


def test_empty_layout(layout_type: embedding.Lattice) -> None:
    coords = get_layout(layout_type=layout_type, n_traps=0)
    check.equal(coords.size(), (0, 2))
