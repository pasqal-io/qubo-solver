from __future__ import annotations

import pandas as pd
import seaborn as sns
import torch

from .data import QUBOSolution
from .qubo_instance import QUBOInstance

__all__ = ["QUBOAnalyzer"]

_BITSTRINGS = "bitstrings"
_COSTS = "costs"
_COUNTS = "counts"
_LABELS = "labels"
_PROBS = "probs"
_GAPS = "gaps"


class QUBOAnalyzer:
    def __init__(
        self,
        solutions: QUBOSolution | list[QUBOSolution],
        labels: str | list[str] | None = None,
    ):
        """
        Analyzer for solutions to a Quadratic Unconstrained Binary Optimization (QUBO) problem.

        Initializes the analyzer with one or a list of QUBOSolutions.

        If a single QUBOSolution is provided, it is automatically wrapped into a list.
        Optionally, you can provide a list of labels corresponding to each QUBOSolution.
        If labels are not provided, they are assigned automatically as '0', '1', etc.

        Args:
            solutions (QUBOSolution | list[QUBOSolution]):
                A single QUBOSolution or a list of QUBOSolution instances.
            labels (str | list[str] | None):
                A list of labels for the QUBOSolutions. Must match the number of solutions.

        Raises:
            ValueError: If no solutions are provided or if the number of labels
                        does not match the number of solutions.
            TypeError: If any solution or label is not of the expected type.
        """
        # Recast solutions into a list if a single solution is provided.
        if not isinstance(solutions, list):
            solutions = [solutions]

        for sol in solutions:
            if not isinstance(sol, QUBOSolution):
                raise TypeError("Each solution must be a QUBOSolution instance.")

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

    @staticmethod
    def tensor_to_bitstrings(bitstring_tensor: torch.Tensor) -> list[str]:
        """
        Converts a torch tensor of bitstrings to a list of bitstring strings.

        Each row in the tensor is assumed to be a bitstring (with integer elements 0 or 1),
        and is converted to a single string (e.g., a row [0, 1, 0, 1, 0, 1] becomes "010101").

        Args:
            bitstring_tensor (torch.Tensor): Tensor of shape (num_bitstrings, bitstring_length)
                                             where each element is an integer (0 or 1).

        Returns:
            list[str]: A list of bitstring strings.
        """
        return ["".join(map(str, row.tolist())) for row in bitstring_tensor]

    @staticmethod
    def bitstrings_to_tensor(bitstring_list: list[str]) -> torch.Tensor:
        """
        Converts a list of bitstring strings to a torch tensor of bitstrings.

        Each bitstring in the list is assumed to be a string of '0's and '1's (e.g., "010101").
        Converts each character to an integer and constructs a tensor of shape
        (num_bitstrings, bitstring_length).

        Args:
            bitstring_list (list[str]): A list of bitstring strings.

        Returns:
            torch.Tensor: A tensor with shape (num_bitstrings, bitstring_length)
                          with integer elements.

        Raises:
            ValueError: If the list is empty or if the bitstrings are not of uniform length.
        """
        if not bitstring_list:
            raise ValueError("The bitstring_list is empty.")
        bit_length = len(bitstring_list[0])
        for bitstr in bitstring_list:
            if len(bitstr) != bit_length:
                raise ValueError("All bitstrings must have the same length.")
        bit_lists = [[int(x) for x in bitstr] for bitstr in bitstring_list]
        return torch.tensor(bit_lists, dtype=torch.int)

    def _solution_to_dataframe(self, solution: QUBOSolution, solution_label: str) -> pd.DataFrame:
        """
        Converts a single QUBOSolution into a pandas DataFrame.
        For better readability, each bitstring is converted to a string representation.

        Args:
            solution (QUBOSolution): The QUBOSolution to convert.
            solution_label (str): The label associated with this solution.

        Returns:
            pd.DataFrame: A DataFrame containing the solution's bitstrings, cost,
                          and optionally counts and probabilities.
        """
        # Convert each row of the bitstring tensor into a string (e.g., "010101").
        bitstring_list = QUBOAnalyzer.tensor_to_bitstrings(solution.bitstrings)
        data = {
            _LABELS: [solution_label] * len(bitstring_list),
            _BITSTRINGS: bitstring_list,
            _COSTS: solution.costs.tolist(),
        }

        if solution.counts is not None:
            data[_COUNTS] = solution.counts.tolist()
        if solution.probabilities is not None:
            data[_PROBS] = solution.probabilities.tolist()
        else:
            tot = sum(data[_COUNTS])
            data[_PROBS] = [x / tot for x in data[_COUNTS]]

        return pd.DataFrame(data)

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Combines all QUBOSolutions into a single DataFrame.
        This DataFrame can be used for filtering, sorting, and analysis.

        Returns:
            pd.DataFrame: The concatenated DataFrame containing all solutions.
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
        """
        Compare two `QUBOSolution` objects and provide a statistical analysis of the differences,
        including degenerate solution matching and mismatch statistics.

        Args:
            target_labels (list[str]): The labels of the solutions to compare. If None, compares
                all solutions.
        """

        def print_diff(
            diff: set[str],
            bs_set: set[str],
            main_label: str,
            compare_label: str,
        ) -> None:
            """
            Prints the differences between two sets of bitstrings.
            Args:
                diff (set[str]): The set of bitstrings that are in main_label but not in
                    compare_label.
                bs_set (set[str]): The set of all unique bitstrings.
                main_label (str): The label of the solution being compared from.
                compare_label (str): The label of the solution being compared to.
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
            raise ValueError("All target labels must be present in the QUBOAnalyzer's labels.")

        # Extract bitstrings for each target label
        bs_list1 = self.df[self.df["labels"] == target_labels[0]]["bitstrings"].tolist()
        bs_list2 = self.df[self.df["labels"] == target_labels[1]]["bitstrings"].tolist()

        # TODO: Once issue about duplicate bitstrings in QUBOSolution is fixed, this can be removed
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
        Returns a DataFrame limited to bitstrings whose probability
        is greater than the provided threshold.

        Args:
            min_probability (float): Minimum probability threshold.
            df (pd.DataFrame | None): DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.

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
        Returns a DataFrame limited to bitstrings whose cost
        is smaller than the provided threshold.

        Args:
            max_cost (float): Maximum cost threshold.
            df (pd.DataFrame | None): DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        if df is None:
            df = self.df

        if _COSTS not in df.columns:
            raise ValueError("No probabilities available in the DataFrame.")

        return df[df[_COSTS] < max_cost]

    def filter_by_percentage(
        self,
        top_percent: float = 1.0,
        column: str = _COSTS,
        order: str = "ascending",
    ) -> pd.DataFrame:
        """
        Returns a DataFrame limited to the best bitstrings
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
            pd.DataFrame: The filtered DataFrame containing, for each solution group, the bitstrings
                          whose cumulative probability (_PROBS)
                        reaches the specified top_percent threshold.

        Raises:
            ValueError: If the specified column is not in the DataFrame,
                        if top_percent is not in (0, 1],
                        or if the order parameter is not "descending" or "ascending".
        """
        df = self.df
        if column not in df.columns:
            raise ValueError(
                f"{column} data is not available. \
                             Please add {column} before filtering."
            )

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
            pd.DataFrame: A DataFrame with each solution label, the average cost over the
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
        Finds all unique bitstrings (with the best cost) in each solution's DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with all unique rows per solution (solution_label)
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

    def calculate_costs(self, Q: QUBOInstance) -> pd.DataFrame:
        """
        Calculates the cost for each bitstring using the provided Q QUBOInstance.

            cost = x^T Q x

        The computed cost is added as the columns _COSTS in the DataFrame.

        Args:
            Q: QUBOInstance

        Returns:
            pd.DataFrame: The updated DataFrame including the _COSTS column.

        Raises:
            ValueError: If a bitstring's length does not match Q.shape[0].
        """

        self.df[_COSTS] = self.df[_BITSTRINGS].apply(Q.evaluate_solution)
        return self.df

    def calculate_gaps(self, opt_cost: float, Q: QUBOInstance | None = None) -> pd.DataFrame:
        """
        Calculates the gaps for each bitstring using the provided optimal cost.
        If costs aren ot present calculates costs as

                cost = x^T Q x

        The computed cost is added as the columns _COSTS in the DataFrame.

        Args:
            Q: QUBOInstance

        Returns:
            pd.DataFrame: The updated DataFrame including the _COSTS column.

        Raises:
            ValueError: If a bitstring's length does not match Q.shape[0].
        """
        if _COSTS in self.df.columns:
            self.df[_GAPS] = abs((self.df[_COSTS] - opt_cost) / opt_cost)
        else:
            if Q is not None:
                self.df[_COSTS] = self.df[_BITSTRINGS].apply(Q.evaluate_solution)
            else:
                self.df[_GAPS] = abs((self.df[_COSTS] - opt_cost) / opt_cost)
        return self.df

    def add_counts(self, counts: list[int] | torch.Tensor) -> None:
        """
        Updates the DataFrame by adding the counts column.

        If counts are provided at a later stage, this method will add the counts
        to the DataFrame and ensure that they match the number of bitstrings.

        Args:
            counts (list[int] | torch.Tensor): A list or tensor of counts.

        Raises:
            ValueError: If the length of counts does not match the number of bitstrings.
        """
        if isinstance(counts, torch.Tensor):
            counts = counts.tolist()  # Convert tensor to list if necessary

        if len(counts) != len(self.df):
            raise ValueError(
                "The number of counts must match" " the number of bitstrings in the DataFrame."
            )

        if _PROBS in self.df.columns:
            # Check if the probabilities are consistent
            # with the counts (probs = counts / total_counts)
            total_counts = sum(self.df[_COUNTS])
            expected_counts = [probs * total_counts for probs in self.df[_PROBS]]
            if not all(abs(p - ep) < 1e-6 for p, ep in zip(counts, expected_counts)):
                raise ValueError("The provided counts do not match probabilities.")

        self.df[_COUNTS] = counts

    def add_probs(self, probs: list[float] | torch.Tensor) -> None:
        """
        Updates the DataFrame by adding the probs column.

        If probs are provided at a later stage, this method will add the probs
        to the DataFrame and ensure that they match the number of bitstrings.

        Args:
            probs (list[float] | torch.Tensor): A list or tensor of probabilities.

        Raises:
            ValueError: If the length of probabilities does not match the number of bitstrings.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.tolist()

        if len(probs) != len(self.df):
            raise ValueError(
                "The number of counts must match" "the number of bitstrings in the DataFrame."
            )

        if _COUNTS in self.df.columns:
            # Check if the probabilities are consistent
            # with the counts (probs = counts / total_counts)
            total_counts = sum(self.df[_COUNTS])
            expected_probs = [count / total_counts for count in self.df[_COUNTS]]
            if not all(abs(p - ep) < 1e-6 for p, ep in zip(probs, expected_probs)):
                raise ValueError("The provided probabilities do not match counts.")

        self.df[_PROBS] = probs

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
            df (pd.DataFrame): The DataFrame to plot. Defaults to None,
                                that means uses self.df.
            y_axis (str): The column name to be plotted on the y-axis.
            sort_by (str | None): Defines the column by which to sort the bitstrings.
                                     If None, no sorting is done.
            sort_order (str): Defines the sorting order. Accepts 'ascending' or 'descending'.
                              Default is 'ascending'. Ignored if `sort_by` is None.

        """
        # Check if the y_axis is available
        if y_axis not in df.columns:
            raise ValueError(
                f"{y_axis} data is not available.\
                              Please add {y_axis} before plotting."
            )
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
            df (pd.DataFrame): The DataFrame to plot. Defaults to None,
                                that means uses self.df.
            x_axis (str): The column name to be plotted on the x-axis.
            y_axis (str): The column name to be plotted on the y-axis.
            sort_by (str | None): Defines the column by which to sort the costs.
                                     If None, no sorting is done.
            sort_order (str): Defines the sorting order. Accepts 'ascending' or 'descending'.
                              Default is 'ascending'. Ignored if `sort_by` is None.
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
            raise ValueError(
                f"{x_axis} data is not available.\
                                Please add {x_axis} before plotting."
            )

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
