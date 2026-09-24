"""QUBO Solver: a library for solving QUBO problems with classical, quantum, and hybrid algorithms.

Solves Quadratic Unconstrained Binary Optimization (QUBO) problems using classical, quantum,
and hybrid algorithms, including on Pasqal neutral-atom QPUs.

Exposes the core data types ([`Instance`][], [`Solution`][], [`Dataset`][], ...), the
[`Solver`][] entry point, and the [`transforms`][], [`embedding`][],
[`drive_shaping`][], and [`solving`][] submodules used to build and run quantum,
hybrid, and classical QUBO solvers.
"""

from __future__ import annotations

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from importlib.metadata import version  # noqa: E402

from pulser.sequence import store_package_version_metadata  # noqa: E402

# isort: split
# qubosolver.types must be imported (and fully initialized) before qubosolver.drive_shaping,
# qubosolver.solver, etc., since those submodules import names back from the qubosolver
# package itself (e.g. `from qubosolver import Instance`). Don't let isort/ruff reorder this
# block alphabetically, or it will reintroduce that circular import.
from qubosolver.types import (  # noqa: E402
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    # Type Aliases
    Bitstring,
    Bitstrings,
    Candidate,
    Dataset,
    Instance,
    LocalEmulator,
    Matrix,
    RemoteEmulator,
    # Classes
    Solution,
    Tensor,
    Vector,
    Vectori,
    # Submodules
    bitstring,
    bitstrings,
    linalg,
    matrix,
    protocols,
    tensor,
    # Functions
    torch_rng,
    vector,
    vectori,
)
from qubosolver.types._checks import _RUNTIME_TYPE_CHECKING  # noqa: E402

# isort: split
from qubosolver import drive_shaping, embedding, solving, transforms  # noqa: E402
from qubosolver.solver import (  # noqa: E402
    ClassicalSolvingConfig,
    DriveShapingConfig,
    EmbeddingConfig,
    QuantumSolvingConfig,
    Solver,
    SolverConfig,
)
from qubosolver.utils import analysis, extract_qubo  # noqa: E402

__all__ = [
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    "Bitstring",
    "Bitstrings",
    "Candidate",
    "ClassicalSolvingConfig",
    "Dataset",
    "DriveShapingConfig",
    "EmbeddingConfig",
    "Instance",
    "LocalEmulator",
    "Matrix",
    "QuantumSolvingConfig",
    "RemoteEmulator",
    "Solution",
    "Solver",
    "SolverConfig",
    "Tensor",
    "Vector",
    "Vectori",
    "analysis",
    "bitstring",
    "bitstrings",
    "drive_shaping",
    "embedding",
    "extract_qubo",
    "linalg",
    "matrix",
    "protocols",
    "solving",
    "tensor",
    "torch_rng",
    "transforms",
    "vector",
    "vectori",
]

__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)

if _RUNTIME_TYPE_CHECKING:
    from beartype import BeartypeConf  # deptry: ignore[DEP004]
    from beartype.claw import beartype_this_package  # deptry: ignore[DEP004]

    beartype_this_package(conf=BeartypeConf(warning_cls_on_decorator_exception=None))
