"""Label types and utilities for QUBO solver.

This module defines flexible labelling mechanisms for QUBO variables and provides
utilities to work with different labelling formats consistently.
"""

from __future__ import annotations


from typing import TypeAlias, Callable, Sequence

Labelling: TypeAlias = Sequence[str] | Callable[[int], str]
"""Type alias for flexible variable labelling in QUBO problems.

Supports two labelling formats:
- Sequence[str]: Labels by index (e.g., ['x', 'y', 'z'])
- Callable[[int], str]: Dynamic labelling function (e.g., lambda i: f'var_{i}')
"""


def _to_callable(labelling: Labelling) -> Callable[[int], str]:
    """Convert any labelling format to a callable function.

    Args:
        labelling: The labelling in any supported format.

    Returns:
        A callable that maps variable indices to string labels.

    Note:
        This is an internal utility function. For Sequence inputs,
        the returned function will raise IndexError if the index
        is not present in the original labelling.
    """
    if callable(labelling):
        return labelling
    else:
        return lambda i: labelling[i]
