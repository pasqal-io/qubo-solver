# Copyright (c) Pasqal
#
# Licensed under the Pasqal Open-Source Software License (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Enum types used by the embedding module."""

from __future__ import annotations

from enum import Enum


class Lattice(Enum):
    """Type of lattice used by the [`greedy_layout`][] embedding algorithm."""

    SQUARE = "square"
    """Arrange qubits on a square lattice grid."""
    TRIANGULAR = "triangular"
    """Arrange qubits on a triangular lattice grid."""
