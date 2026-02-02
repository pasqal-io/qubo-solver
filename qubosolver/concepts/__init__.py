from __future__ import annotations

"""
CONCEPTS & STRUCTURAL TYPING IN PYTHON
======================================

This module defines the project's 'Concepts' using Python's Structural Typing
system (PEP 544).

Unlike Nominal Typing (C++/Java), where a class must explicitly inherit from
an interface, Python's Protocol system allows for "Static Duck Typing."

Key Principles:
---------------
1. BEHAVIOR OVER LINEAGE:
   A class is compatible with a Protocol if it shares the same 'structure'
   (method names and signatures), regardless of its inheritance tree.

2. EXPLICIT SIGNATURES:
   Mypy validates the entire method contract:
   - Argument names and count (including 'self').
   - Type hints for parameters.
   - Return type compatibility.

3. THE ELLIPSIS (...) CONVENTION:
   Protocols use '...' in method bodies to indicate they are stubs,
   equivalent to 'virtual void func() = 0;' in C++.

Usage Example:
--------------
    from typing import Protocol

    class Speaker(Protocol):
        def talk(self, message: str) -> str: ...

    def process(entity: Speaker) -> None:
        print(entity.talk("Hello"))

    # Any class implementing `talk(self, message: str) -> str`
    # will be accepted by Mypy here.


To test this concept statically:
-------------------------------
In your test files, use:
    def test_contract() -> None:
        _: Toto = MyImplementation() # Mypy will fail if signature mismatches
"""

from .backend import Backend

__all__ = [
    "Backend",
]
