"""QUBO Solver: a library for building, solving, and analysing QUBO instances.

Exposes the core data types (`Instance`, `Solution`, `Dataset`, ...), the
`Solver` entry point, and the `transforms`, `embedding`, `drive_shaping`, and
`solvers` submodules used to build and run quantum and classical QUBO solvers.
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
    # Qubo* TypeAliases
    QuboSolution,
    QuboSingleSolution,
    QuboInstance,
    QuboDataset,
    # Deprecated QUBO* classes
    QUBOSolution,
    QUBOInstance,
    QUBODataset,
)
from qubosolver.utils import extract_qubo, analysis

from importlib.metadata import version
from pulser.sequence import store_package_version_metadata

from qubosolver import transforms, drive_shaping, embedding, solvers

from qubosolver.solvers import Solver, QuboSolver

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
    "solvers",
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
    "Solver",
    "Dataset",
    "LocalEmulator",
    "RemoteEmulator",
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    # Functions
    "torch_rng",
    "extract_qubo",
    # Qubo* TypeAliases
    "QuboSolution",
    "QuboSingleSolution",
    "QuboInstance",
    "QuboDataset",
    "QuboSolver",
    # Deprecated QUBO* classes
    "QUBOSolution",
    "QUBOInstance",
    "QUBODataset",
]

__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)

if _RUNTIME_TYPE_CHECKING:
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package(conf=BeartypeConf(warning_cls_on_decorator_exception=None))
