from __future__ import annotations

from typing import TYPE_CHECKING
from qubosolver.types._checks import _RUNTIME_TYPE_CHECKING

if _RUNTIME_TYPE_CHECKING and not TYPE_CHECKING:
    from beartype.typing import Protocol as Protocol
else:
    from typing import Protocol as Protocol

__all__ = [
    "Protocol",
]
