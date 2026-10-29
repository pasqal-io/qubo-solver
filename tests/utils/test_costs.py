# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import pytest

from qubosolver import bitstrings, matrix
from qubosolver.utils import _costs


def test_quadratic_cost_rejects_batched_input() -> None:
    with pytest.raises(ValueError, match="Use batched_quadratic_cost instead"):
        _costs.quadratic_cost(bitstrings.zeros(2, 2), matrix.zeros(2))
