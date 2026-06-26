from __future__ import annotations

import os
import typing

# Tensor Types
_RUNTIME_TYPE_CHECKING: bool = os.getenv("QUBO_SOLVER_RUNTIME_CHECKS", "0") == "1"

TYPE_CHECKING = typing.TYPE_CHECKING or _RUNTIME_TYPE_CHECKING

_T = typing.TypeVar("_T")


def debug_runtime_typecheck(target: _T) -> _T:
    """Applies @beartype if QUBO_SOLVER_RUNTIME_CHECKS is enabled, otherwise a no-op."""
    if _RUNTIME_TYPE_CHECKING:
        from beartype import beartype

        return beartype(target)  # type: ignore[no-any-return]
    return target
