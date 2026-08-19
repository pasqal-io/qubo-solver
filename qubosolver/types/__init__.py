from __future__ import annotations

from qubosolver.types import (
    bitstring,
    bitstrings,
    matrix,
    tensor,
    vector,
    vectori,
    linalg,
    protocols,
)
from qubosolver.types.linalg import Bitstring, Bitstrings, Matrix, Tensor, Vector, Vectori
from qubosolver.types.solution import Solution, SingleSolution
from qubosolver.types.instance import Instance
from qubosolver.types.dataset import Dataset
from qubosolver.types.backends import (
    LocalEmulator,
    RemoteEmulator,
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
)
from qubosolver.types.random import torch_rng
from qubosolver.types.aliases import (
    # Deprecated QUBO* classes
    QUBOSolution,
    QUBOInstance,
    QUBODataset,
    # Qubo* TypeAliases
    QuboSolution,
    QuboSingleSolution,
    QuboInstance,
    QuboDataset,
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
    # Qubo* TypeAliases
    "QuboSolution",
    "QuboSingleSolution",
    "QuboInstance",
    "QuboDataset",
    # Deprecated QUBO* classes
    "QUBOSolution",
    "QUBOInstance",
    "QUBODataset",
]
