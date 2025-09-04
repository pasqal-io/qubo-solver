from __future__ import annotations

from importlib import import_module

from .data import *
from .qubo_instance import *
from .qubo_types import *

list_of_submodules = [".data", ".utils", ".qubo_instance", ".pipeline"]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(
        import_module(submodule, package="qubosolver"), "__all__"
    )
    __all__ += __all_submodule__
