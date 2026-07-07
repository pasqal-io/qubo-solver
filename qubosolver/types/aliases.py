from __future__ import annotations

from typing_extensions import deprecated, TypeAlias

from .solution import Solution, SingleSolution
from .analyzer import Analyzer
from .instance import Instance
from .dataset import Dataset


# ---------------------------------------------------------------------------
# Deprecated QUBO* classes
# ---------------------------------------------------------------------------

@deprecated("Use `qubosolver.Solution` instead")
class QUBOSolution(Solution):
    """Legacy solution collection class.

    !!! warning "Deprecated"
        This class is deprecated and will be removed in a future version.
        Use [`qubosolver.Solution`][] instead.
    """


@deprecated("Use `qubosolver.Analyzer` instead")
class QUBOAnalyzer(Analyzer):
    """Legacy solution analyser class.

    !!! warning "Deprecated"
        This class is deprecated and will be removed in a future version.
        Use [`qubosolver.Analyzer`][] instead.
    """


@deprecated("Use `qubosolver.Instance` instead")
class QUBOInstance(Instance):
    """Legacy QUBO problem instance class.

    !!! warning "Deprecated"
        This class is deprecated and will be removed in a future version.
        Use [`qubosolver.Instance`][] instead.
    """


@deprecated("Use `qubosolver.Dataset` instead")
class QUBODataset(Dataset):
    """Legacy QUBO dataset class.

    !!! warning "Deprecated"
        This class is deprecated and will be removed in a future version.
        Use [`qubosolver.Dataset`][] instead.
    """


# ---------------------------------------------------------------------------
# Qubo* TypeAliases
# ---------------------------------------------------------------------------

QuboSolution: TypeAlias = Solution
"""Alias for [`Solution`][qubosolver.Solution]."""


QuboSingleSolution: TypeAlias = SingleSolution
"""Alias for [`SingleSolution`][qubosolver.SingleSolution]."""


QuboAnalyzer: TypeAlias = Analyzer
"""Alias for [`Analyzer`][qubosolver.Analyzer]."""


QuboInstance: TypeAlias = Instance
"""Alias for [`Instance`][qubosolver.Instance]."""


QuboDataset: TypeAlias = Dataset
"""Alias for [`Dataset`][qubosolver.Dataset]."""
