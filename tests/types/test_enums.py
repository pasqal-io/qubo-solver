from __future__ import annotations

import pytest_check as check

from qubosolver.solving import classical


def test_str_enum_type() -> None:
    enum_type: classical.Algorithm = classical.Algorithm.TABU_SEARCH
    check.is_instance(enum_type, classical.Algorithm)
    check.is_instance(enum_type, str)

    str_type: str = "tabu_search"
    check.is_not_instance(str_type, classical.Algorithm)
    check.is_instance(str_type, str)

    check.equal(enum_type, str_type)

    invalid_str_type: str = "invalid_type"
    check.is_not_instance(invalid_str_type, classical.Algorithm)
    check.is_instance(invalid_str_type, str)
