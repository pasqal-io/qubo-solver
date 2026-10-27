# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Greedy embedding algorithm and its layout helpers."""

from __future__ import annotations

from .greedy import Greedy
from .layout import get_layout

__all__ = [
    "Greedy",
    "get_layout",
]
