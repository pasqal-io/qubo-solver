"""Free functions for analysing QUBO solutions.

Converts one or more [`qubosolver.Solution`][] objects into a unified
`pandas.DataFrame`, for filtering, comparing, and summarizing solver outputs.

Example:
    ```python
    df = to_dataframe([sol_a, sol_b], labels=["classical", "quantum"])
    ```
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
import pandas as pd
import seaborn as sns

from qubosolver import Solution, bitstrings

_BITSTRINGS = "bitstrings"
_COSTS = "costs"
_COUNTS = "counts"
_LABELS = "labels"
_PROBS = "probs"
_GAPS = "gaps"


def _solution_to_dataframe(solution: Solution, solution_label: str) -> pd.DataFrame:
    """Convert a single `Solution` into a pandas `DataFrame`.

    Each bitstring is converted to a string representation for readability.

    Args:
        solution: The Solution to convert.
        solution_label: The label associated with this solution.

    Returns:
        A `DataFrame` containing the solution's bitstrings, costs,
            counts, and probabilities.
    """
    solution.check_consistency(instance=None, throw=True)

    # Convert each row of the bitstring tensor into a string (e.g., "010101").
    bitstring_list = bitstrings.to_strings(solution.bitstrings)
    data = {
        _LABELS: [solution_label] * len(solution),
        _BITSTRINGS: bitstring_list,
        _COSTS: solution.costs.tolist(),
        _COUNTS: solution.counts.tolist(),
        _PROBS: solution.probabilities.tolist(),
    }
    return pd.DataFrame(data)


def to_dataframe(
    solutions: Sequence[Solution],
    *,
    labels: Sequence[str] | Literal["auto"] = "auto",
) -> pd.DataFrame:
    """Convert one or more Solutions into a single, unified `DataFrame`.

    The resulting `DataFrame` can be used for filtering, sorting, and analysis.

    Args:
        solutions: A list of [`qubosolver.Solution`][].
        labels: One label per solution used to identify each group in the
            `DataFrame`. Defaults to ``"0"``, ``"1"``, … when ``"auto"``.

    Returns:
        The concatenated `DataFrame` containing all solutions.

    Raises:
        ValueError: If the number of labels does not match the number of solutions.
    """
    if labels == "auto":
        labels = [str(i) for i in range(len(solutions))]
    elif len(labels) != len(solutions):
        raise ValueError("The number of labels must equal the number of QUBOSolutions provided.")

    df_list = []
    df_list.append(_solution_to_dataframe(Solution(), solution_label=""))
    for label, sol in zip(labels, solutions):
        df_list.append(_solution_to_dataframe(sol, solution_label=label))
    return pd.concat(df_list, ignore_index=True)


def _filter_by_percentage(
    df: pd.DataFrame,
    *,
    top_percent: float = 1.0,
    column: str = _COSTS,
    order: Literal["ascending", "descending"] = "ascending",
) -> pd.DataFrame:
    """Return the best-ranked rows of each solution group up to a cumulative probability.

    Rows in each solution group (grouped by label) are sorted by `column`
    (ascending if `order` is "ascending", i.e. lower values are better;
    descending otherwise) and kept until the cumulative probability of the
    selected rows would reach `top_percent`.

    Args:
        df: `DataFrame` to filter.
        top_percent: A threshold between 0 and 1 representing the fraction of
            cumulative probability. For example, 0.1 means select bitstrings
            until their cumulative probability is >= 10%.
        column: The column by which to sort the rows (e.g. `_COSTS`, `_GAPS`,
            or `_PROBS`). Defaults to `_COSTS`.
        order: Either "ascending" or "descending", see above.

    Returns:
        The filtered `DataFrame` containing, for each solution
            group, the bitstrings whose cumulative probability (`_PROBS`)
            reaches the specified `top_percent` threshold.

    Raises:
        ValueError: If the specified column is not in the `DataFrame`,
            or if top_percent is not in (0, 1].
    """
    if column not in df.columns:
        raise ValueError(f"{column} data is not available. \
                         Please add {column} before filtering.")

    if not (0 < top_percent <= 1):
        raise ValueError("top_percent must be a float between 0 and 1.")

    sorted_df = df.sort_values(by=column, ascending=(order == "ascending"))
    # Cumulative probability of all *previous* rows in each group.
    grouped_probs = sorted_df.groupby(_LABELS)[_PROBS]
    cum_probs_before = grouped_probs.shift(fill_value=0).groupby(sorted_df[_LABELS]).cumsum()
    # Keep every row whose group hasn't yet reached top_percent before it.
    return sorted_df[cum_probs_before < top_percent]


def _average_cost(df: pd.DataFrame, *, top_percent: float = 1.0) -> pd.DataFrame:
    """Compute the average cost over the best `top_percent` bitstrings of each solution.

    Args:
        df: `DataFrame` to compute over.
        top_percent: A fraction between 0 and 1 representing the fraction
            of lowest-cost bitstrings to consider.

    Returns:
        A `DataFrame` with each solution label, the average cost over the
            best top_percent bitstrings, and the count of bitstrings used.
    """
    df_top = _filter_by_percentage(df, top_percent=top_percent)
    results = []
    for label, group in df_top.groupby(_LABELS):
        avg_cost = group[_COSTS].mean()
        results.append(
            {
                _LABELS: label,
                "average cost": avg_cost,
                "bitstrings considered": len(group),
            }
        )

    return pd.DataFrame(results)


def _best_bitstrings(df: pd.DataFrame, *, atol: float = 0.0, rtol: float = 0.0) -> pd.DataFrame:
    """Find all unique bitstrings with the best (lowest) cost in each solution group.

    Args:
        df: `DataFrame` to compute over.
        atol: Absolute tolerance used when comparing costs to the minimum.
        rtol: Relative tolerance used when comparing costs to the minimum.

    Returns:
        A `DataFrame` with all unique rows per solution (solution_label)
            that have the best (lowest) cost.
    """
    best_list = []
    for _, sol in df.groupby(_LABELS):
        min_cost = sol[_COSTS].min()
        # Filter all rows with the cost equal to the minimum cost in this group
        best = sol[np.isclose(sol[_COSTS], min_cost, atol=atol, rtol=rtol)]
        # Optionally, drop duplicate bitstring entries (if bitstrings are duplicated)
        best = best.drop_duplicates(subset=[_BITSTRINGS])
        best_list.append(best)
    best_rows = pd.concat(best_list, ignore_index=True)
    return best_rows


def _add_gaps(df: pd.DataFrame, *, opt_cost: float) -> pd.DataFrame:
    """Compute the optimality gap for each bitstring and add it as a `gaps` column.

    The gap is computed as $|cost - c^*| / |c^*|$, where $c^*$ is `opt_cost`.

    Args:
        df: `DataFrame` to update.
        opt_cost: The known optimal cost used to compute the gap.

    Returns:
        A new `DataFrame` including the gaps column.
    """
    df = df.copy()
    df[_GAPS] = abs((df[_COSTS] - opt_cost) / opt_cost)
    return df


## PLOTTING ROUTINES
def _plot_vs_bitstrings(
    df: pd.DataFrame,
    y_axis: str,
    sort_by: str | None = None,
    sort_order: str = "descending",
    context: str = "notebook",
) -> sns.axisgrid.FacetGrid:
    """Plot a bar chart of costs, counts, or probabilities as a function of bitstrings.

    Args:
        df: The `DataFrame` to plot.
        y_axis: The column name to be plotted on the y-axis.
        sort_by: The column by which to sort the bitstrings. If None, no
            sorting is done.
        sort_order: Either "ascending" or "descending". Default is
            "descending". Ignored if `sort_by` is None.
        context: Seaborn plotting context (e.g. "notebook", "talk").

    Returns:
        The resulting plot.
    """
    # Check if the y_axis is available
    if y_axis not in df.columns:
        raise ValueError(f"{y_axis} data is not available.\
                          Please add {y_axis} before plotting.")
    if sort_by and sort_by not in df.columns:
        raise ValueError(f"{sort_by} is not a valid column for sorting.")

    if sort_by == y_axis:
        df = df.pivot_table(
            index=_BITSTRINGS,
            columns=_LABELS,
            values=y_axis,
            fill_value=0,
        ).reset_index()
        df = df.melt(id_vars=_BITSTRINGS, var_name=_LABELS, value_name=y_axis)
        df = df.sort_values(by=sort_by, ascending=(sort_order == "ascending"))
    else:
        df = df.pivot_table(
            index=[_BITSTRINGS, sort_by],
            columns=_LABELS,
            values=y_axis,
            fill_value=0,
        ).reset_index()
        df = df.melt(id_vars=[_BITSTRINGS, sort_by], var_name=_LABELS, value_name=y_axis)
        df = df.sort_values(by=sort_by, ascending=(sort_order == "ascending"))

    # Set color palette
    cmap = sns.color_palette("viridis", n_colors=len(df[_LABELS].unique().tolist()))

    with sns.plotting_context(context):
        g = sns.catplot(
            data=df,
            x=_BITSTRINGS,
            y=y_axis,
            hue=_LABELS,
            kind="bar",
            order=df[_BITSTRINGS].unique().tolist(),
            height=6,
            aspect=1.5,
            palette=cmap,
        )

    g.set_axis_labels(_BITSTRINGS, y_axis)

    g.set_xticklabels(rotation=90)
    return g


def _plot_no_bitstrings(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    sort_by: str | None = None,
    sort_order: str = "ascending",
    context: str = "notebook",
) -> sns.axisgrid.FacetGrid:
    """Plot a bar chart of probabilities or counts as a function of cost.

    Args:
        df: The `DataFrame` to plot.
        x_axis: Column name for the x-axis (e.g. ``"costs"``, ``"gaps"``).
        y_axis: Column name for the y-axis (e.g. ``"probs"``, ``"counts"``).
        sort_by: Column by which to order the x-axis values before plotting.
            Must be one of *x_axis* or *y_axis*.  No sorting when *None*.
        sort_order: ``"ascending"`` or ``"descending"``.  Default is
            ``"ascending"``.  Ignored when *sort_by* is *None*.
        context: Seaborn plotting context (e.g. ``"notebook"``, ``"talk"``).

    Returns:
        The resulting grouped bar chart, one bar
            group per unique x-axis value with hue mapped to solution labels.

    Raises:
        ValueError: If *x_axis* or *y_axis* is not a column in *df*, or if
            *sort_by* is not one of *x_axis* or *y_axis*.
    """
    if x_axis not in df.columns:
        raise ValueError(f"{x_axis} data is not available. Please add {x_axis} before plotting.")

    if y_axis not in df.columns:
        raise ValueError(f"{y_axis} data is not available. Please add {y_axis} before plotting.")

    if sort_by:
        if sort_by not in [x_axis, y_axis]:
            raise ValueError(f"{sort_by} is not a valid column for sorting.")

    df = df.groupby([_LABELS, x_axis], as_index=False).agg({y_axis: "sum"})
    df = df.pivot_table(
        index=x_axis,
        columns=_LABELS,
        values=y_axis,
        fill_value=0,
    ).reset_index()
    df = df.melt(id_vars=x_axis, var_name=_LABELS, value_name=y_axis)
    df = df.sort_values(by=sort_by, ascending=(sort_order == "ascending"))

    # Set color palette
    cmap = sns.color_palette("viridis", n_colors=len(df[_LABELS].unique().tolist()))

    with sns.plotting_context(context):
        g = sns.catplot(
            data=df,
            x=x_axis,
            y=y_axis,
            hue=_LABELS,
            kind="bar",
            order=df[x_axis].unique().tolist(),
            height=6,
            aspect=1.5,  # This ensures the bars are side by side
            palette=cmap,
        )

    # Set axis labels
    g.set_axis_labels(x_axis, y_axis)

    return g


def _plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    *,
    labels: list[str] | None = None,
    sort_by: str | None = None,
    sort_order: str = "ascending",
    probability_threshold: float | None = None,
    cost_threshold: float | None = None,
    top_percent: float | None = None,
    context: str = "notebook",
) -> sns.axisgrid.FacetGrid:
    """Plot costs, counts, or probabilities as a function of bitstrings or of cost.

    Filters `df` by `labels`, `probability_threshold`, `cost_threshold`, and
    `top_percent` (each applied only if not None), then dispatches to
    `_plot_vs_bitstrings` when `x_axis` is `_BITSTRINGS`, or to
    `_plot_no_bitstrings` otherwise.

    Args:
        df: The `DataFrame` to plot.
        x_axis: Column name for the x-axis.
        y_axis: Column name for the y-axis.
        labels: If given, restrict the plot to these solution labels.
        sort_by: Column by which to sort before plotting. Forwarded to the
            underlying plot function.
        sort_order: Either "ascending" or "descending". Default is "ascending".
        probability_threshold: If given, keep only rows with probability
            strictly greater than this value.
        cost_threshold: If given, keep only rows with cost strictly lower
            than this value.
        top_percent: If given, keep only the best rows per group up to this
            cumulative probability (see `_filter_by_percentage`).
        context: Seaborn plotting context (e.g. "notebook", "talk").

    Returns:
        The resulting plot.

    Raises:
        ValueError: If `x_axis` is not a column in `df`.
    """
    df = df.copy()

    if x_axis not in df.columns:
        raise ValueError(f"{x_axis} data is not available.\
                            Please add {x_axis} before plotting.")

    if labels:
        df = df[df[_LABELS].isin(labels)]

    if probability_threshold is not None:
        df = df[df[_PROBS] > probability_threshold]

    if cost_threshold is not None:
        df = df[df[_COSTS] < cost_threshold]

    if top_percent is not None:
        df = _filter_by_percentage(df, top_percent)

    if x_axis == _BITSTRINGS:
        g = _plot_vs_bitstrings(
            df=df,
            y_axis=y_axis,
            sort_by=sort_by,
            sort_order=sort_order,
            context=context,
        )
        return g
    else:
        g = _plot_no_bitstrings(
            df=df,
            x_axis=x_axis,
            y_axis=y_axis,
            sort_by=sort_by,
            sort_order=sort_order,
            context=context,
        )
        return g
