# Copyright (c) Pasqal
#
# Licensed under the Pasqal Open-Source Software License (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

from qubosolver._checks import _RUNTIME_TYPE_CHECKING

if _RUNTIME_TYPE_CHECKING and not TYPE_CHECKING:
    from beartype.typing import Protocol as Protocol  # deptry: ignore[DEP004]
else:
    from typing import Protocol as Protocol

__all__ = [
    "Protocol",
]
