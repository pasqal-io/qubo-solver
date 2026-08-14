from __future__ import annotations

from typing import Any
from abc import ABC
from dataclasses import fields


class _Config(ABC):
    """Abstract base class for all solver configuration dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain dict.

        Subclasses that only expose a subset of fields relevant to the
        active algorithm override this method.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def field_names(cls) -> set[str]:
        """Return the set of declared field names for this config class."""
        return {f.name for f in fields(cls)}  # type: ignore[arg-type]
