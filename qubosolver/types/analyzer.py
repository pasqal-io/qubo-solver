"""Analysis utilities for QUBO solutions.

Provides :class:`Analyzer`, which aggregates one or more
:class:`~qubosolver.types.solution.Solution` objects into a unified
:class:`~pandas.DataFrame` and exposes filtering, statistical, and plotting
helpers for comparing solver outputs.

Typical usage:

    analyzer = Analyzer([sol_a, sol_b], labels=["classical", "quantum"])
    analyzer.calculate_gaps(opt_cost=-42.0)
    analyzer.plot(x_axis="bitstrings", y_axis="costs", top_percent=0.1)
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .solution import Solution
from .instance import Instance
from . import bitstrings
from .linalg import Vectori, Vector

_BITSTRINGS = "bitstrings"
_COSTS = "costs"
_COUNTS = "counts"
_LABELS = "labels"
_PROBS = "probs"
_GAPS = "gaps"


class Analyzer:
    """Aggregates and analyses one or more QUBO solutions.

    Converts [`qubosolver.Solution`][] objects into a
    unified `pandas.DataFrame` (``self.df``) with columns for
    bitstrings, costs, and optionally counts, probabilities, and gaps.
    Multiple solutions can be labelled and compared side-by-side through the
    filtering and plotting helpers.

    Args:
        solutions: A single [`qubosolver.Solution`][]
            or a list of them.  A bare instance is automatically wrapped in a list.
        labels: One label per solution used to identify each group in the
            `DataFrame` and plots.  Defaults to ``"0"``, ``"1"``, … when `None`.

    Raises:
        ValueError: If the number of labels does not match the number of solutions.
        TypeError: If any element of *solutions* is not a
             [`qubosolver.Solution`][], or if any label
            is not a `str`.
    """

    def __init__(
        self,
        solutions: Solution | list[Solution],
        labels: str | list[str] | None = None,
    ):
        # Recast solutions into a list if a single solution is provided.
        if not isinstance(solutions, list):
            solutions = [solutions]

        for sol in solutions:
            if not isinstance(sol, Solution):
                raise TypeError("Each solution must be a Solution instance.")

        self.solutions = solutions

        # Validate labels if provided.
        if labels is not None:
            # Recast labels into a list if a single solution is provided.
            if not isinstance(labels, list):
                labels = [labels]

            if len(labels) != len(solutions):
                raise ValueError(
                    "The number of labels must equal the number of QUBOSolutions provided."
                )
            for label in labels:
                if not isinstance(label, str):
                    raise TypeError("Each label must be a string.")
            self.labels = labels
        else:
            self.labels = [str(i) for i in range(len(solutions))]

        self.df = self._to_dataframe()

    def _solution_to_dataframe(self, solution: Solution, solution_label: str) -> pd.DataFrame:
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
        # Convert each row of the bitstring tensor into a string (e.g., "010101").
        bitstring_list = bitstrings.to_strings(solution.bitstrings)
        data = {
            _LABELS: [solution_label] * len(bitstring_list),
            _BITSTRINGS: bitstring_list,
            _COSTS: solution.costs.tolist(),
        }

        if solution.counts.numel() > 0:
            data[_COUNTS] = solution.counts.tolist()
        if solution.probabilities.numel() > 0:
            data[_PROBS] = solution.probabilities.tolist()
        else:
            if solution.counts.numel() > 0:
                tot = sum(data[_COUNTS])
                data[_PROBS] = [x / tot for x in data[_COUNTS]]

        return pd.DataFrame(data)

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Combines all QUBOSolutions into a single `DataFrame`.
        This `DataFrame` can be used for filtering, sorting, and analysis.

        Returns:
            pd.DataFrame: The concatenated `DataFrame` containing all solutions.
        """
        df_list = []
        # Construct DataFrames for each solution using their associated label.
        for label, sol in zip(self.labels, self.solutions):
            df_list.append(self._solution_to_dataframe(sol, solution_label=label))
        return pd.concat(df_list, ignore_index=True)

    def compare_qubo_solutions(
        self,
        target_labels: list[str],
    ) -> None:
        """Compare the bitstring sets of exactly two labelled solutions.

        Prints a human-readable summary to *stdout* reporting:

        - The total and unique bitstring counts for each solution.
        - Bitstrings present in the first solution but absent from the second,
          and vice-versa.
        - The ratio of differing bitstrings over the total unique set.

        Note:
            Duplicate bitstrings within a single solution are deduplicated
            before comparison.  This is a temporary workaround until the
            upstream duplicate-bitstring issue in
            [`Solution`][qubosolver.Solution] is resolved.

        Args:
            target_labels: Exactly two labels identifying the solutions to
                compare.  Both must be present in the [`Analyzer`][] labels.

        Raises:
            ValueError: If `len(target_labels) != 2`, or if any label is not
                present in the [`Analyzer`][] labels.
        """

        def print_diff(
            diff: set[str],
            bs_set: set[str],
            main_label: str,
            compare_label: str,
        ) -> None:
            """Print the bitstrings present in one solution but absent from the other.

            Args:
                diff: Bitstrings in `main_label` not present in `compare_label`.
                bs_set: Full set of unique bitstrings for `main_label`.
                main_label: Label of the solution being compared from.
                compare_label: Label of the solution being compared to.
            """
            if len(diff) > 0:
                print(f"\nBitstrings in {main_label} not present in {compare_label}:")
                for bs in diff:
                    print("-", bs)
                print(
                    f"\nRatio of different bitstrings: {len(diff)}/{len(bs_set)} = "
                    + f"{(len(diff)/len(bs_set))*100:.0f}%"
                )

        # Validate target labels
        if len(target_labels) != 2:
            raise ValueError("Exactly two target labels must be provided for comparison.")
        if not all(label in self.labels for label in target_labels):
            raise ValueError("All target labels must be present in the Analyzer's labels.")

        # Extract bitstrings for each target label
        bs_list1 = self.df[self.df["labels"] == target_labels[0]]["bitstrings"].tolist()
        bs_list2 = self.df[self.df["labels"] == target_labels[1]]["bitstrings"].tolist()

        # TODO: Once issue about duplicate bitstrings in Solution is fixed, this can be removed
        bs_set1 = set(bs_list1)
        bs_set2 = set(bs_list2)

        print(
            f"Comparing two lists of bitstrings:\n1. {target_labels[0]}: {len(bs_list1)} bitstrings"
            + f" ({len(bs_set1)} unique strings)\n2. {target_labels[1]}: {len(bs_list2)} bitstrings"
            + f" ({len(bs_set2)} unique strings)"
        )

        # Analyze differences
        diff1 = bs_set1 - bs_set2
        diff2 = bs_set2 - bs_set1

        if len(diff1) == 0 and len(diff2) == 0:
            print("\nThe lists contain exactly the same bitstrings.")
            return
        else:
            print_diff(diff1, bs_set1, target_labels[0], target_labels[1])
            print_diff(diff2, bs_set2, target_labels[1], target_labels[0])

    def filter_by_probability(
        self, min_probability: float, df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Returns a `DataFrame` limited to bitstrings whose probability
        is greater than the provided threshold.

        Args:
            min_probability (float): Minimum probability threshold.
            df (pd.DataFrame | None): `DataFrame` to filter.

        Returns:
            pd.DataFrame: The filtered `DataFrame`.

        Raises:
            ValueError: If the 'probabilities' column is not present.
        """

        if df is None:
            df = self.df

        if _PROBS not in df.columns:
            raise ValueError("No probabilities available in the DataFrame.")
        return df[df[_PROBS] > min_probability]

    def filter_by_cost(self, max_cost: float, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Returns a `DataFrame` limited to bitstrings whose cost
        is smaller than the provided threshold.

        Args:
            max_cost (float): Maximum cost threshold.
            df (pd.DataFrame | None): `DataFrame` to filter.

        Returns:
            pd.DataFrame: The filtered `DataFrame`.
        """

        if df is None:
            df = self.df

        if _COSTS not in df.columns:
            raise ValueError("No costs available in the DataFrame.")

        return df[df[_COSTS] < max_cost]

    def filter_by_percentage(
        self,
        top_percent: float = 1.0,
        column: str = _COSTS,
        order: str = "ascending",
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
                        if top_percent is not in (0, 1],
                        or if the order parameter is not "descending" or "ascending".
        """
        df = self.df
        if column not in df.columns:
            raise ValueError(f"{column} data is not available. \
                             Please add {column} before filtering.")

        if not (0 < top_percent <= 1):
            raise ValueError("top_percent must be a float between 0 and 1.")

        if order not in ("ascending", "descending"):
            raise ValueError("The keep parameter must be either 'ascending' or 'descending'.")

        filtered_list = []
        for label, group in df.groupby(_LABELS):
            # Sort the group based on the specified column using the desired order.
            sorted_group = group.sort_values(by=column, ascending=(order == "ascending"))
            cumulative = 0.0
            selected_indices = []
            # Use the _PROBS column to accumulate probability
            for idx, row in sorted_group.iterrows():
                cumulative += row[_PROBS]
                selected_indices.append(idx)
                if cumulative >= top_percent:
                    break

            filtered_group = sorted_group.loc[selected_indices]
            filtered_list.append(filtered_group)
        return pd.concat(filtered_list, ignore_index=True)

    def average_cost(self, top_percent: float = 1) -> pd.DataFrame:
        """
        Calculates the average cost for the best top_percent of bitstrings (lowest cost)
        for each solution.

        Args:
            top_percent (float): A fraction between 0 and 1 representing the percentage
                                 of lowest cost bitstrings to consider.

        Returns:
            pd.DataFrame: A `DataFrame` with each solution label, the average cost over the
                          best top_percent bitstrings, and the count of bitstrings used.
        """
        df_top = self.filter_by_percentage(top_percent)
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

    def best_bitstrings(self) -> pd.DataFrame:
        """
        Finds all unique bitstrings (with the best cost) in each solution's `DataFrame`.

        Returns:
            pd.DataFrame: A `DataFrame` with all unique rows per solution (solution_label)
                          that have the best (lowest) cost.
        """
        best_list = []
        for label, sol in self.df.groupby(_LABELS):
            min_cost = sol[_COSTS].min()
            # Filter all rows with the cost equal to the minimum cost in this group
            best = sol[sol[_COSTS] == min_cost]
            # Optionally, drop duplicate bitstring entries (if bitstrings are duplicated)
            best = best.drop_duplicates(subset=[_BITSTRINGS])
            best_list.append(best)
        best_rows = pd.concat(best_list, ignore_index=True)
        return best_rows

    def calculate_costs(self, instance: Instance) -> pd.DataFrame:
        """
        Calculates the cost for each bitstring using the provided instance.

            cost = x^T Q x

        The computed cost is added as the columns _COSTS in the `DataFrame`.

        Args:
            instance: Instance

        Returns:
            pd.DataFrame: The updated `DataFrame` including the _COSTS column.

        Raises:
            ValueError: If a bitstring's length does not match instance.shape[0].
        """

        self.df[_COSTS] = self.df[_BITSTRINGS].apply(instance.evaluate_solution)
        return self.df

    def calculate_gaps(self, opt_cost: float, instance: Instance = Instance()) -> pd.DataFrame:
        """
        Calculates the gaps for each bitstring using the provided optimal cost.
        If costs are not present, calculates costs as ``x^T Q x`` first.

        The computed gaps are added as the ``gaps`` column in the `DataFrame`.

        Args:
            opt_cost (float): The known optimal cost used to compute
                ``|cost - opt_cost| / |opt_cost|``.
            instance (Instance): Optional QUBO instance used to compute
                costs if they are not already present in the `DataFrame`.

        Returns:
            pd.DataFrame: The updated `DataFrame` including the gaps column.
        """
        if _COSTS in self.df.columns:
            self.df[_GAPS] = abs((self.df[_COSTS] - opt_cost) / opt_cost)
        else:
            if instance.size > 0:
                self.df[_COSTS] = self.df[_BITSTRINGS].apply(instance.evaluate_solution)
            else:
                self.df[_GAPS] = abs((self.df[_COSTS] - opt_cost) / opt_cost)
        return self.df

    def add_counts(self, counts: Vectori) -> None:
        """
        Updates the `DataFrame` by adding the counts column.

        If counts are provided at a later stage, this method will add the counts
        to the `DataFrame` and ensure that they match the number of bitstrings.

        Args:
            counts (Vectori): An ``int64`` tensor of counts.

        Raises:
            ValueError: If the length of counts does not match the number of bitstrings,
                or if the counts are inconsistent with existing probabilities.
        """
        counts_l = counts.tolist()  # Convert tensor to list if necessary

        if len(counts_l) != len(self.df):
            raise ValueError(
                "The number of counts must match" " the number of bitstrings in the DataFrame."
            )

        if _PROBS in self.df.columns:
            # Check if the probabilities are consistent
            # with the counts (probs = counts / total_counts)
            total_counts = sum(self.df[_COUNTS])
            expected_counts = [probs * total_counts for probs in self.df[_PROBS]]
            if not all(abs(p - ep) < 1e-6 for p, ep in zip(counts_l, expected_counts)):
                raise ValueError("The provided counts do not match probabilities.")

        self.df[_COUNTS] = counts_l

    def add_probs(self, probs: Vector) -> None:
        """
        Updates the `DataFrame` by adding the probs column.

        If probs are provided at a later stage, this method will add the probs
        to the `DataFrame` and ensure that they match the number of bitstrings.

        Args:
            probs (Vector): A float tensor of probabilities.

        Raises:
            ValueError: If the length of probabilities does not match the number of bitstrings,
                or if the probabilities are inconsistent with existing counts.
        """
        probs_l = probs.tolist()

        if len(probs_l) != len(self.df):
            raise ValueError(
                "The number of counts must match" "the number of bitstrings in the DataFrame."
            )

        if _COUNTS in self.df.columns:
            # Check if the probabilities are consistent
            # with the counts (probs = counts / total_counts)
            total_counts = sum(self.df[_COUNTS])
            expected_probs = [count / total_counts for count in self.df[_COUNTS]]
            if not all(abs(p - ep) < 1e-6 for p, ep in zip(probs_l, expected_probs)):
                raise ValueError("The provided probabilities do not match counts.")

        self.df[_PROBS] = probs_l

    ## PLOTTING ROUTINES
    @staticmethod
    def plot_vs_bitstrings(
        df: pd.DataFrame,
        y_axis: str,
        sort_by: str | None = None,
        sort_order: str = "descending",
        context: str = "notebook",
    ) -> sns.axisgrid.FacetGrid:
        """
        Plots a bar chart of costs, counts, or probabilities as a function of bitstrings.

        Args:
            df (pd.DataFrame): The `DataFrame` to plot. Defaults to None,
                                that means uses self.df.
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

    @staticmethod
    def plot_no_bitstrings(
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
            raise ValueError(
                f"{x_axis} data is not available. Please add {x_axis} before plotting."
            )

        if y_axis not in df.columns:
            raise ValueError(
                f"{y_axis} data is not available. Please add {y_axis} before plotting."
            )

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
        self,
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
        df = self.df.copy()

        if x_axis not in df.columns:
            raise ValueError(f"{x_axis} data is not available.\
                                Please add {x_axis} before plotting.")

        if labels:
            df = df[df[_LABELS].isin(labels)]

        if probability_threshold is not None:
            df = self.filter_by_probability(probability_threshold, df)

        if cost_threshold is not None:
            df = self.filter_by_cost(cost_threshold, df)

        if top_percent is not None:
            df = self.filter_by_percentage(top_percent)

        if x_axis == _BITSTRINGS:
            g = self.plot_vs_bitstrings(
                df=df,
                y_axis=y_axis,
                sort_by=sort_by,
                sort_order=sort_order,
                context=context,
            )
            return g
        else:
            g = self.plot_no_bitstrings(
                df=df,
                x_axis=x_axis,
                y_axis=y_axis,
                sort_by=sort_by,
                sort_order=sort_order,
                context=context,
            )
            return g
