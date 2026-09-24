"""Core data types for the QUBO solver: tensors, solutions, and backends.

This subpackage re-exports the tensor submodules (`bitstring`, `bitstrings`,
`matrix`, `tensor`, `vector`, `vectori`, `linalg`), the container classes
(`Solution`, `Candidate`, `Instance`, `Dataset`), and the emulator backend
wrappers used throughout the public API.
"""

from __future__ import annotations

from qubosolver.types import (
    bitstring,
    bitstrings,
    linalg,
    matrix,
    protocols,
    tensor,
    vector,
    vectori,
)
from qubosolver.types.backends import (
    AutoLocalEmulatorBackend,
    AutoRemoteEmulatorBackend,
    LocalEmulator,
    RemoteEmulator,
)
from qubosolver.types.dataset import Dataset
from qubosolver.types.instance import Instance
from qubosolver.types.linalg import Bitstring, Bitstrings, Matrix, Tensor, Vector, Vectori
from qubosolver.types.random import torch_rng
from qubosolver.types.solution import Candidate, Solution

__all__ = [
    "AutoLocalEmulatorBackend",
    "AutoRemoteEmulatorBackend",
    "Bitstring",
    "Bitstrings",
    "Candidate",
    "Dataset",
    "Instance",
    "LocalEmulator",
    "Matrix",
    "RemoteEmulator",
    "Solution",
    "Tensor",
    "Vector",
    "Vectori",
    "bitstring",
    "bitstrings",
    "linalg",
    "matrix",
    "protocols",
    "tensor",
    "torch_rng",
    "vector",
    "vectori",
]
