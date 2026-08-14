"""Free functions for analysing QUBO solutions.

Convert one or more :class:`~qubosolver.types.solution.Solution` objects into
a unified :class:`~pandas.DataFrame` and provides filtering, statistical, and
plotting helpers for comparing solver outputs.

Typical usage:

    df = to_dataframe([sol_a, sol_b], labels=["classical", "quantum"])
    df = add_gaps(df, opt_cost=-42.0)
    plot(df, x_axis="bitstrings", y_axis="costs", top_percent=0.1)
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
    """
    Converts a single Solution into a pandas `DataFrame`.
    For better readability, each bitstring is converted to a string representation.

    Args:
        solution (Solution): The Solution to convert.
        solution_label (str): The label associated with this solution.

    Returns:
        pd.DataFrame: A `DataFrame` containing the solution's bitstrings, cost,
                      and optionally counts and probabilities.
    """
    solution.check_consistency(instance=None, throw=True)

    if not solution:
        return pd.DataFrame()

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
    """
    Converts one or more Solutions into a single, unified `DataFrame`.
    This `DataFrame` can be used for filtering, sorting, and analysis.

    Args:
        solutions: A list of [`qubosolver.Solution`][].
        labels: One label per solution used to identify each group in the
            `DataFrame`. Defaults to ``"0"``, ``"1"``, … when ``"auto"``.

    Returns:
        pd.DataFrame: The concatenated `DataFrame` containing all solutions.

    Raises:
        ValueError: If the number of labels does not match the number of solutions.
    """
    if labels == "auto":
        labels = [str(i) for i in range(len(solutions))]
    elif len(labels) != len(solutions):
        raise ValueError("The number of labels must equal the number of QUBOSolutions provided.")

    df_list = []
    for label, sol in zip(labels, solutions):
        df_list.append(_solution_to_dataframe(sol, solution_label=label))
    return pd.concat(df_list, ignore_index=True)


def filter_by_percentage(
    df: pd.DataFrame,
    *,
    top_percent: float = 1.0,
    column: str = _COSTS,
    order: Literal["ascending", "descending"] = "ascending",
) -> pd.DataFrame:
    """
    Returns a `DataFrame` limited to the best bitstrings
    in a given column for each solution group,
    where "best" means that the cumulative probability (_PROBS)
    of the selected rows reaches at least
    top_percent. The sorting order is controlled by the
    `order` parameter: if "ascending", the group is sorted
    in ascending order (lower values are considered better);
    if "descending", sorted in descending order.

    Args:
        df (pd.DataFrame): `DataFrame` to filter.
        top_percent (float): A threshold between 0 and 1 representing
                             the fraction of cumulative probability.
                             For example, 0.1 means select bitstrings
                             until their cumulative probability is ≥ 10%.
        column (str): The key (column) by which to sort the rows
                                (e.g. _COSTS, _GAPS, or _PROBS).
                                Defaults to _COSTS.
        order (str): Either "ascending" or "descending". If "ascending",
                     rows are sorted in ascending order (lower values are better).
                     If "descending", rows are sorted in descending order
                     (higher values are better).

    Returns:
        pd.DataFrame: The filtered `DataFrame` containing, for each solution group, the bitstrings
                      whose cumulative probability (_PROBS)
                    reaches the specified top_percent threshold.

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


def average_cost(df: pd.DataFrame, *, top_percent: float = 1.0) -> pd.DataFrame:
    """
    Calculates the average cost for the best top_percent of bitstrings (lowest cost)
    for each solution.

    Args:
        df (pd.DataFrame): `DataFrame` to compute over.
        top_percent (float): A fraction between 0 and 1 representing the percentage
                             of lowest cost bitstrings to consider.

    Returns:
        pd.DataFrame: A `DataFrame` with each solution label, the average cost over the
                      best top_percent bitstrings, and the count of bitstrings used.
    """
    df_top = filter_by_percentage(df, top_percent)
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


def best_bitstrings(df: pd.DataFrame, *, atol: float = 0.0, rtol: float = 0.0) -> pd.DataFrame:
    """
    Finds all unique bitstrings (with the best cost) in each solution's `DataFrame`.

    Args:
        df (pd.DataFrame): `DataFrame` to compute over.
        atol (float): Absolute tolerance used when comparing costs to the minimum.
        rtol (float): Relative tolerance used when comparing costs to the minimum.

    Returns:
        pd.DataFrame: A `DataFrame` with all unique rows per solution (solution_label)
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

def add_gaps(df: pd.DataFrame, *, opt_cost: float) -> pd.DataFrame:
    """
    Calculates the gaps for each bitstring using the provided optimal cost.

    The computed gaps are added as the ``gaps`` column in the returned `DataFrame`.

    Args:
        df: `DataFrame` to update.
        opt_cost (float): The known optimal cost used to compute
            ``|cost - opt_cost| / |opt_cost|``.

    Returns:
        pd.DataFrame: A new `DataFrame` including the gaps column.
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
    """
    Plots a bar chart of costs, counts, or probabilities as a function of bitstrings.

    Args:
        df (pd.DataFrame): The `DataFrame` to plot.
        y_axis (str): The column name to be plotted on the y-axis.
        sort_by (str | None): Defines the column by which to sort the bitstrings.
                                 If None, no sorting is done.
        sort_order (str): Defines the sorting order. Accepts 'ascending' or 'descending'.
                          Default is 'descending'. Ignored if ``sort_by`` is None.
        context (str): Seaborn plotting context (e.g. ``"notebook"``, ``"talk"``).

    Returns:
        sns.axisgrid.FacetGrid: The resulting plot.
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
    """
    Plots a bar chart of probabilities or counts as a function of cost.

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
        sns.axisgrid.FacetGrid: The resulting grouped bar chart, one bar
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


def plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    labels: list[str] | None = None,
    sort_by: str | None = None,
    sort_order: str = "ascending",
    probability_threshold: float | None = None,
    cost_threshold: float | None = None,
    top_percent: float | None = None,
    context: str = "notebook",
) -> sns.axisgrid.FacetGrid:
    """
    A wrapper function that chooses between plotting costs, counts, or probabilities
    as a function of bitstrings or as a function of cost.
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
        df = filter_by_percentage(df, top_percent)

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
