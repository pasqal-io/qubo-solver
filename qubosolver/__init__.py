from __future__ import annotations

from importlib import import_module

from .data import *  # noqa: F403
from .qubo_instance import *  # noqa: F403
from .qubo_types import *  # noqa: F403
from .config import *  # noqa: F403
from .utils import *  # noqa: F403

from importlib.metadata import version
from pulser.sequence import store_package_version_metadata

list_of_submodules = [".data", ".utils", ".qubo_instance", ".config"]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(import_module(submodule, package="qubosolver"), "__all__")
    __all__ += __all_submodule__


__version__ = version("qubo-solver")
store_package_version_metadata("qubosolver", __version__)
