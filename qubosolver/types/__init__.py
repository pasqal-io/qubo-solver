"""Core data types for the QUBO solver: tensors, solutions, and backends.

This subpackage re-exports the tensor submodules (`bitstring`, `bitstrings`,
`matrix`, `tensor`, `vector`, `vectori`, `linalg`), the container classes
(`Solution`, `SingleSolution`, `Instance`, `Dataset`), and the emulator backend
wrappers used throughout the public API.
"""

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
]
