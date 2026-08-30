"""QUBO Solver: a library for solving Quadratic Unconstrained Binary Optimization (QUBO) problems using
classical, quantum, and hybrid algorithms, including on Pasqal neutral-atom QPUs.

Exposes the core data types ([`Instance`][], [`Solution`][], [`Dataset`][], ...), the
[`Solver`][] entry point, and the [`transforms`][], [`embedding`][],
[`drive_shaping`][], and [`solvers`][] submodules used to build and run quantum,
hybrid, and classical QUBO solvers.
"""

from __future__ import annotations

from qubosolver.types._checks import _RUNTIME_TYPE_CHECKING

from qubosolver.types import (
    # Submodules
    bitstring,
    bitstrings,
    matrix,
    tensor,
    vector,
    vectori,
    linalg,
    protocols,
    # Type Aliases
    Bitstring,
    Bitstrings,
    Matrix,
    Tensor,
    Vector,
    Vectori,
    # Classes
    Solution,
    SingleSolution,
    Instance,
    Dataset,
    LocalEmulator,
    RemoteEmulator,
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    # Functions
    torch_rng,
)
from qubosolver.utils import extract_qubo, analysis

from importlib.metadata import version
from pulser.sequence import store_package_version_metadata

from qubosolver import transforms, drive_shaping, embedding, solving

from qubosolver.solver import (
    Solver,
    SolverConfig,
    DecompositionConfig,
    DriveShapingConfig,
    EmbeddingConfig,
    ClassicalSolvingConfig,
    QuantumSolvingConfig,
)


__all__ = [
    # Submodules
    "bitstring",
    "bitstrings",
    "matrix",
    "tensor",
    "vector",
    "vectori",
    "linalg",
    "protocols",
    "solving",
    "transforms",
    "embedding",
    "drive_shaping",
    "analysis",
    # Type Aliases
    "Bitstring",
    "Bitstrings",
    "Matrix",
    "Tensor",
    "Vector",
    "Vectori",
    # Classes
    "Solution",
    "SingleSolution",
    "Instance",
    "Dataset",
    "LocalEmulator",
    "RemoteEmulator",
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    # Functions
    "torch_rng",
    "extract_qubo",
    # Config-based API
    "Solver",
    "SolverConfig",
    "DecompositionConfig",
    "DriveShapingConfig",
    "EmbeddingConfig",
    "ClassicalSolvingConfig",
    "QuantumSolvingConfig",
]

__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)

if _RUNTIME_TYPE_CHECKING:
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package(conf=BeartypeConf(warning_cls_on_decorator_exception=None))
