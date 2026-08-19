from __future__ import annotations

import pytest
import pytest_check as check

from qubosolver import (
    # Canonical classes
    Solution,
    SingleSolution,
    Instance,
    Dataset,
    # Qubo* aliases
    QuboSolution,
    QuboSingleSolution,
    QuboInstance,
    QuboDataset,
    # Deprecated QUBO* classes
    QUBOSolution,
    QUBOInstance,
    QUBODataset,
)

# ---------------------------------------------------------------------------
# Qubo* TypeAlias tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, canonical",
    [
        (QuboSolution, Solution),
        (QuboSingleSolution, SingleSolution),
        (QuboInstance, Instance),
        (QuboDataset, Dataset),
    ],
    ids=[
        "QuboSolution",
        "QuboSingleSolution",
        "QuboInstance",
        "QuboDataset",
    ],
)
def test_qubo_alias_is_canonical(alias: type, canonical: type) -> None:
    """Each Qubo* alias must resolve to the same type as the canonical class."""
    check.is_(alias, canonical)


# ---------------------------------------------------------------------------
# Deprecated QUBO* class tests – each triggers a DeprecationWarning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deprecated_cls, canonical, match_msg",
    [
        (QUBOSolution, Solution, "Use `qubosolver.Solution` instead"),
        (QUBOInstance, Instance, "Use `qubosolver.Instance` instead"),
    ],
    ids=["QUBOSolution", "QUBOInstance"],
)
def test_deprecated_class_warns(deprecated_cls: type, canonical: type, match_msg: str) -> None:
    """Instantiating a QUBO* deprecated class must emit a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match=match_msg):
        obj = deprecated_cls()
    check.is_instance(obj, canonical)


def test_qubo_dataset_wrong_case_deprecation() -> None:
    """QUBODataset requires a coefficients tensor and must emit a DeprecationWarning."""
    import torch

    coefficients = torch.zeros(2, 2, 1)
    with pytest.warns(DeprecationWarning, match="Use `qubosolver.Dataset` instead"):
        dataset = QUBODataset(coefficients)
    check.is_instance(dataset, Dataset)
