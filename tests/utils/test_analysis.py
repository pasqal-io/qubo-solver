from __future__ import annotations

import pandas as pd
import pytest

from qubosolver import analysis


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "labels": ["a", "a", "a", "b", "b"],
            "costs": [3, 1, 2, 5, 4],
            "probs": [0.1, 0.5, 0.4, 0.6, 0.4],
        }
    )


@pytest.mark.parametrize(
    ("top_percent", "order", "expected_costs"),
    [
        (0.05, "ascending", [1, 4]),
        (0.05, "descending", [3, 5]),
        (0.1, "descending", [3, 5]),  # exercises float-precision edge (0.5 - 0.4 != 0.1 exactly)
        (0.3, "descending", [2, 3, 5]),
        (0.5, "ascending", [1, 4, 5]),
        (0.9, "ascending", [1, 2, 4, 5]),
        (1.0, "ascending", [1, 2, 3, 4, 5]),
    ],
)
def test_filter_by_percentage(
    top_percent: float, order: str, expected_costs: list[int]
) -> None:
    result = analysis.filter_by_percentage(_df(), top_percent=top_percent, order=order)
    assert sorted(result["costs"].tolist()) == expected_costs


def test_filter_by_percentage_keeps_all_rows_when_group_never_reaches_threshold() -> None:
    df = pd.DataFrame(
        {
            "labels": ["a", "b", "b"],
            "costs": [1, 2, 3],
            "probs": [1.0, 0.2, 0.2],
        }
    )
    result = analysis.filter_by_percentage(df, top_percent=0.9)
    assert sorted(result["costs"].tolist()) == [1, 2, 3]


def test_filter_by_percentage_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        analysis.filter_by_percentage(_df(), column="gaps")


def test_filter_by_percentage_invalid_top_percent_raises() -> None:
    with pytest.raises(ValueError):
        analysis.filter_by_percentage(_df(), top_percent=0.0)
    with pytest.raises(ValueError):
        analysis.filter_by_percentage(_df(), top_percent=1.5)
