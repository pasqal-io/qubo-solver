from __future__ import annotations

import pytest
import pytest_check as check

from qubosolver import Instance, matrix
from qubosolver.embedding import blade


def test_empty_embedding() -> None:
    instance = Instance(matrix.zeros(0))
    with pytest.raises(ValueError, match="empty instance"):
        blade.embed(instance)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_single_atom_embedding(value: float) -> None:
    instance = Instance(matrix.zeros(1).fill_(value))
    register = blade.embed(instance)
    check.equal(len(register), 1)
