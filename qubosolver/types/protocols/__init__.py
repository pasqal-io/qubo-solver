"""Structural protocols ("Concepts") for the QUBO solver, using PEP 544 Protocol typing.

Unlike nominal typing (C++/Java), where a class must explicitly inherit from an
interface, Python's `Protocol` allows "static duck typing": a class is compatible
with a protocol if it implements the same methods with matching signatures,
regardless of its inheritance tree. mypy validates the full method contract
(argument names, types, and return type); when runtime type checking is enabled,
`isinstance` checks against the protocol are also supported (see
[`Protocol`][qubosolver.types.protocols._protocol.Protocol]).

Example:
    ```python
    from typing import Protocol

    class Speaker(Protocol):
        def talk(self, message: str) -> str: ...

    def process(entity: Speaker) -> None:
        print(entity.talk("Hello"))

    # Any class implementing `talk(self, message: str) -> str`
    # is accepted by mypy here, without inheriting from Speaker.
    ```
"""

from __future__ import annotations

from .backend import Backend

__all__ = [
    "Backend",
]
