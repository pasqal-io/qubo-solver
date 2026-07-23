"""Runtime type-checking helpers controlled by the ``QUBO_SOLVER_RUNTIME_CHECKS`` env var.

Set ``QUBO_SOLVER_RUNTIME_CHECKS=1`` before launching the process to enable
beartype-powered runtime type validation across the library.  When the variable
is absent or set to any other value, all decorators defined here are no-ops and
add zero overhead.

Module-level constants
----------------------
_RUNTIME_TYPE_CHECKING : bool
    ``True`` when ``QUBO_SOLVER_RUNTIME_CHECKS=1`` is set in the environment.
TYPE_CHECKING : bool
    ``True`` during static analysis (``typing.TYPE_CHECKING``) **or** when
    runtime checking is active.  Use this instead of ``typing.TYPE_CHECKING``
    when you need imports that are required both by the type-checker and by
    beartype at runtime.
"""

from __future__ import annotations

import os
import typing

_RUNTIME_TYPE_CHECKING: bool = os.getenv("QUBO_SOLVER_RUNTIME_CHECKS", "0") == "1"

TYPE_CHECKING = typing.TYPE_CHECKING or _RUNTIME_TYPE_CHECKING

_T = typing.TypeVar("_T")


def debug_runtime_typecheck(target: _T) -> _T:
    """Decorator that enables full beartype runtime type-checking on *target*.

    When ``QUBO_SOLVER_RUNTIME_CHECKS=1`` is set, wraps *target* with
    `beartype.beartype` using its default (strictest) strategy so that
    every argument and return value is validated on each call.

    When runtime checks are disabled this is a transparent no-op: *target* is
    returned unchanged with no performance impact.

    Args:
        target: A callable (function, method, or class) to decorate.

    Returns:
        The beartype-wrapped callable when runtime checks are active,
        or *target* itself otherwise.

    Example::

        @debug_runtime_typecheck
        def solve(matrix: np.ndarray) -> Solution: ...
    """
    if _RUNTIME_TYPE_CHECKING:
        from beartype import beartype

        return beartype(target)  # type: ignore[no-any-return]
    return target


def no_runtime_typecheck(target: _T) -> _T:
    """Decorator that explicitly **disables** beartype runtime type-checking on *target*.

    Applies :class:`~beartype.BeartypeStrategy.O0` (zero-cost passthrough) so
    that *target* is exempt from type validation even when
    ``QUBO_SOLVER_RUNTIME_CHECKS=1`` is active globally.  Use this on hot paths
    or on callables whose signatures cannot be validated by beartype.

    When runtime checks are disabled entirely this is also a no-op.

    Args:
        target: A callable (function, method, or class) to exempt from checking.

    Returns:
        The beartype O0-wrapped callable when runtime checks are active,
        or *target* itself otherwise.

    Example::

        @no_runtime_typecheck
        def _fast_inner_loop(data: list) -> None: ...
    """
    if _RUNTIME_TYPE_CHECKING:
        from beartype import beartype, BeartypeConf, BeartypeStrategy

        return beartype(target, conf=BeartypeConf(strategy=BeartypeStrategy.O0))  # type: ignore[no-any-return]
    return target
