# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Core data types for the QUBO solver: bitstrings, solutions, and backends.

This subpackage re-exports the bitstring submodules (`bitstring`, `bitstrings`), the
container classes (`Solution`, `Candidate`, `Instance`, `Dataset`), and the emulator
backend wrappers used throughout the public API.
"""

from __future__ import annotations

from qubosolver.types import (
    bitstring,
    bitstrings,
    protocols,
)
from qubosolver.types.backends import (
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.types.dataset import Dataset
from qubosolver.types.instance import Instance
from qubosolver.types.random import numpy_rng
from qubosolver.types.solution import Candidate, Solution

__all__ = [
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    "Candidate",
    "Dataset",
    "Instance",
    "LocalEmulator",
    "RemoteEmulator",
    "Solution",
    "bitstring",
    "bitstrings",
    "numpy_rng",
    "protocols",
]
