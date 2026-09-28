from __future__ import annotations

from importlib import import_module

from .data import *  # noqa: F403
from .qubo_instance import *  # noqa: F403
from .qubo_types import *  # noqa: F403

list_of_submodules = [".data", ".utils", ".qubo_instance", ".pipeline"]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(import_module(submodule, package="qubosolver"), "__all__")
    __all__ += __all_submodule__
