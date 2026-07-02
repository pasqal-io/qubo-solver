from __future__ import annotations

from qubosolver.types import bitstring, bitstrings, matrix, tensor, vector, vectori, linalg
from qubosolver.types.linalg import Bitstring, Bitstrings, Matrix, Tensor, Vector, Vectori
from qubosolver.types.solution import QUBOSolution, QUBOSingleSolution
from qubosolver.types.analyzer import QUBOAnalyzer
from qubosolver.types.instance import QUBOInstance
from qubosolver.types.dataset import QUBODataset
from qubosolver.types.enums import (
    EmbedderType,
    LayoutType,
    DriveType,
    DensityType,
    ClassicalSolverType,
)
from qubosolver.types.backends import (
    LocalEmulator,
    RemoteEmulator,
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
)
from qubosolver.types.random import torch_rng

__all__ = [
    # Submodules
    "bitstring",
    "bitstrings",
    "matrix",
    "tensor",
    "vector",
    "vectori",
    "linalg",
    # Type Aliases
    "Bitstring",
    "Bitstrings",
    "Matrix",
    "Tensor",
    "Vector",
    "Vectori",
    # Classes
    "QUBOSolution",
    "QUBOSingleSolution",
    "QUBOAnalyzer",
    "QUBOInstance",
    "QUBODataset",
    "LocalEmulator",
    "RemoteEmulator",
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    # Enums
    "EmbedderType",
    "LayoutType",
    "DriveType",
    "DensityType",
    "ClassicalSolverType",
    # Functions
    "torch_rng",
]
