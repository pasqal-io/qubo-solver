# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""QUBO Solver: a library for solving QUBO problems with classical, quantum, and hybrid algorithms.

Solves Quadratic Unconstrained Binary Optimization (QUBO) problems using classical, quantum,
and hybrid algorithms, including on Pasqal neutral-atom QPUs.

Exposes the core data types ([`Instance`][], [`Solution`][], [`Dataset`][], ...) and the
[`transforms`][], [`embedding`][], [`drive_shaping`][], and [`solving`][] submodules used to
build and run quantum, hybrid, and classical QUBO solvers.
"""

from __future__ import annotations

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# isort: split
# qubosolver.types must be imported (and fully initialized) before qubosolver.drive_shaping,
# qubosolver.solving, etc., since those submodules import names back from the qubosolver
# package itself (e.g. `from qubosolver import Instance`). Don't let isort/ruff reorder this
# block alphabetically, or it will reintroduce that circular import.
from qubosolver.types import (  # noqa: E402
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    Candidate,
    Dataset,
    Instance,
    LocalEmulator,
    RemoteEmulator,
    # Classes
    Solution,
    # Submodules
    bitstring,
    bitstrings,
    # Functions
    numpy_rng,
    protocols,
)

# isort: split
from qubosolver import drive_shaping, embedding, solving, transforms  # noqa: E402
from qubosolver.utils import analysis, extract_qubo  # noqa: E402

__all__ = [
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    "Candidate",
    "Dataset",
    "Instance",
    "LocalEmulator",
    "RemoteEmulator",
    "Solution",
    "analysis",
    "bitstring",
    "bitstrings",
    "drive_shaping",
    "embedding",
    "extract_qubo",
    "numpy_rng",
    "protocols",
    "solving",
    "transforms",
]

from importlib.metadata import version  # noqa: E402

from pulser.sequence import store_package_version_metadata  # noqa: E402

__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)
