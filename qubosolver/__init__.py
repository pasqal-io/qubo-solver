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
    Analyzer,
    Instance,
    Dataset,
    LocalEmulator,
    RemoteEmulator,
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    # Enums
    EmbedderType,
    DriveType,
    LayoutType,
    DensityType,
    ClassicalSolverType,
    # Functions
    torch_rng,
    # Qubo* TypeAliases
    QuboSolution,
    QuboSingleSolution,
    QuboAnalyzer,
    QuboInstance,
    QuboDataset,
    # Deprecated QUBO* classes
    QUBOSolution,
    QUBOAnalyzer,
    QUBOInstance,
    QUBODataset,
)
from qubosolver.config import (
    SolverConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    ClassicalConfig,
    DecompositionConfig,
)

from importlib.metadata import version
from pulser.sequence import store_package_version_metadata

from qubosolver import solvers, transforms, drive_shaping, embedding
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
    "solvers",
    "transforms",
    "embedding",
    "drive_shaping",
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
    "Analyzer",
    "Instance",
    "Solver",
    "Dataset",
    "LocalEmulator",
    "RemoteEmulator",
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    # Enums
    "EmbedderType",
    "DriveType",
    "LayoutType",
    "DensityType",
    "ClassicalSolverType",
    # Configs
    "SolverConfig",
    "EmbeddingConfig",
    "DriveShapingConfig",
    "ClassicalConfig",
    "DecompositionConfig",
    # Functions
    "torch_rng",
    # Qubo* TypeAliases
    "QuboSolution",
    "QuboSingleSolution",
    "QuboAnalyzer",
    "QuboInstance",
    "QuboDataset",
    "QuboSolver",
    # Deprecated QUBO* classes
    "QUBOSolution",
    "QUBOAnalyzer",
    "QUBOInstance",
    "QUBODataset",
]

__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)

if _RUNTIME_TYPE_CHECKING:
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package(conf=BeartypeConf(warning_cls_on_decorator_exception=None))
