from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, cast

import dimod
import numpy as np
import torch
from dwave.preprocessing.lower_bounds import roof_duality

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.config import SolverConfig
from qubosolver.qubo_types import SolutionStatusType


def bit_flip_local_search(
    qubo_func: Callable[[np.ndarray], float], s: np.ndarray, shuffle: bool = True
) -> tuple[np.ndarray, float]:
    """
    Performs a local search by flipping bits to improve the objective value.

    Args:
        qubo_func: Function that computes the objective value for a solution array.
        s (np.ndarray): Binary array representing a candidate solution.
        shuffle (bool, optional): Shuffle to diversify

    Returns:
        tuple[np.ndarray, float]: The improved solution and its objective value.
    """
    s_current = s.copy()
    current_objective = qubo_func(s_current)
    while True:
        best_idx = None
        best_obj = current_objective
        indices = list(range(len(s_current)))
        if shuffle:
            np.random.shuffle(indices)  # option to diversify
        # Evaluate all possible flips, keep best
        for i in indices:
            s_new = s_current.copy()
            s_new[i] = 1 - s_new[i]
            new_objective = qubo_func(s_new)
            if new_objective < best_obj:
                best_obj = new_objective
                best_idx = i
        # If no improvements, stop
        if best_idx is None:
            break
        # Apply best flip
        s_current[best_idx] = 1 - s_current[best_idx]
        current_objective = best_obj
    return s_current, current_objective


def hansen_fixing(qubo: QUBOInstance) -> Dict[int, int]:
    """
    Identifies and fixes variables in a QUBO instance based on threshold conditions.

    This method determines whether a variable should be fixed to 0 or 1 by computing
    lower and upper bounds from the diagonal and off-diagonal elements of the QUBO matrix.

    Args:
        qubo (QUBOInstance): The QUBO instance containing the coefficients matrix.

    Returns:
        Dict[int, int]: A dictionary mapping variable indices to fixed values (0 or 1).
    """
    if qubo.coefficients is None:
        raise ValueError("QUBO coefficients are not initialized.")

    fixed_dict: Dict[int, int] = {}
    size_raw = qubo.size
    size: int = cast(int, size_raw)
    epsilon: float = 1e-8  # Tolerance to avoid floating-point precision issues

    for i in range(size):
        ci = qubo.coefficients[i, i].item()  # Diagonal element

        q_minus = sum(min(0, qubo.coefficients[i, j].item()) for j in range(size) if j != i)
        q_plus = sum(max(0, qubo.coefficients[i, j].item()) for j in range(size) if j != i)

        if ci + q_minus * 2 >= -epsilon:
            fixed_dict[i] = 0
        elif ci + q_plus * 2 <= epsilon:
            fixed_dict[i] = 1

    return fixed_dict


def dwave_roof_duality_fixing(qubo_inst: QUBOInstance) -> Dict[int, int]:
    """
    Applies D-Wave's roof duality method to identify and fix variables in a QUBO instance.

    Roof duality provides a way to determine certain variable values with certainty
    without solving the entire optimization problem.

    Args:
        qubo_inst (QUBOInstance): The QUBO instance to be processed.

    Returns:
        Dict[int, int]: A dictionary mapping fixed variable indices to their values.
    """
    if qubo_inst.coefficients is None:
        raise ValueError("QUBO coefficients are not initialized.")

    # Convert QUBO matrix to a Binary Quadratic Model (BQM) format
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_inst.coefficients.cpu().numpy())

    # Apply roof duality to identify fixable variables
    _, raw_fixed = roof_duality(bqm, strict=True)

    # Reconstruire un Dict[int, int] pour satisfaire MyPy
    fixed_variables: Dict[int, int] = {int(var): int(val) for var, val in raw_fixed.items()}
    return fixed_variables


class Fixtures:
    """
    Handles all preprocessing and postprocessing logic for QUBO problems.

    This class centralizes the transformation or validation of the problem instance
    before solving and modification or annotation of the solution after solving.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        """
        Initialize the fixture handler with the QUBO instance and solver configuration.

        Args:
            instance (QUBOInstance): The QUBO problem instance to process.
            config (SolverConfig): Solver configuration, which may include flags for enabling
                postprocessing (e.g., do_postprocessing).
        """
        self.instance = instance
        self.config = config

        self.reduced_qubo: QUBOInstance = deepcopy(instance)
        self.fixation_rule_list: List[Callable[[QUBOInstance], Dict[int, int]]] = [
            hansen_fixing,
            dwave_roof_duality_fixing,
        ]
        self.fixed_var_dict_list: List[Dict[int, int]] = []

    @property
    def n_fixed_variables(self) -> int:
        """Returns the number of fixed variables.

        Returns:
            int: The number of fixed variables.
        """
        return sum([len(fixed) for fixed in self.fixed_var_dict_list])

    def preprocess(self) -> QUBOInstance:
        """
        Apply preprocessing steps to the QUBO instance before solving.

        Returns:
            QUBOInstance: The processed or annotated instance.
        """

        # Check if postprocessing is enabled via the configuration.
        if not hasattr(self.config, "do_postprocessing") or not self.config.do_postprocessing:
            return self.instance

        # Apply every rules until exhaustion
        self.apply_full_fixation_exhaust()

        return self.instance

    def postprocess(self, solution: QUBOSolution) -> QUBOSolution:
        """
        Apply postprocessing steps to the QUBO solution after solving.

        This method iterates over all solutions in the bitstrings tensor and, for each solution,
        performs a local bit-flip search to attempt to improve its objective value. The objective
        is computed using the QUBOInstance's evaluate_solution method.

        Args:
            solution (QUBOSolution): The raw solution from the solver.

        Returns:
            QUBOSolution: The updated solution with improved bitstrings and costs.
        """

        if not hasattr(self.config, "do_postprocessing") or not self.config.do_postprocessing:
            return solution

        # If there are no bitstrings, return the solution unchanged.
        if solution.bitstrings.numel() == 0:
            return solution

        # Define an objective function that uses the existing evaluate_solution method.
        def qubo_objective(s_arr: np.ndarray) -> float:
            # Convert the solution array to a list of integers
            return self.instance.evaluate_solution(s_arr.tolist())

        improved_bitstrings = []
        improved_costs = []
        num_solutions = solution.bitstrings.shape[0]

        for idx in range(num_solutions):
            # Get the current solution (row) as a numpy array of integers.
            s_orig = solution.bitstrings[idx].detach().cpu().numpy().astype(int)
            # Apply bit-flip local search to improve the solution.
            s_improved, new_cost = bit_flip_local_search(qubo_objective, s_orig)
            improved_bitstrings.append(s_improved)
            improved_costs.append(new_cost)

        # Create new tensors for the improved solutions and their costs.
        new_bitstrings_tensor = torch.tensor(np.array(improved_bitstrings), dtype=torch.float32)
        new_costs_tensor = torch.tensor(improved_costs, dtype=torch.float32)

        # Update the solution object.
        solution.bitstrings = new_bitstrings_tensor
        solution.costs = new_costs_tensor

        if self.config.do_preprocessing and self.config.do_postprocessing:
            solution.solution_status = SolutionStatusType.PREPOSTPROCESSED
        else:
            solution.solution_status = SolutionStatusType.POSTPROCESSED

        return solution

    def reduce_qubo(self, fixed_dict: Dict[int, int]) -> None:
        """
        Applies variable fixation to reduce the size of the QUBO problem.

        This function modifies the QUBO coefficient matrix by:
        - Removing rows and columns corresponding to fixed variables.
        - Adjusting diagonal elements to account for fixed variables.

        Args:
            fixed_dict (Dict[int, int]): A dictionary of fixed variable assignments.
                - Keys are variable indices.
                - Values are fixed binary values (0 or 1).

        Returns:
            None: Modifies `self.reduced_qubo` in place.
        """
        Q = self.reduced_qubo.coefficients.clone()

        fixed_to_0 = {i for i, v in fixed_dict.items() if v == 0}
        fixed_to_1 = {i for i, v in fixed_dict.items() if v == 1}
        fixed_vars = sorted(fixed_to_0 | fixed_to_1, reverse=True)

        for i in fixed_vars:
            if i >= Q.shape[0]:
                continue

            if i in fixed_to_1:
                for j in range(Q.shape[0]):
                    if j != i:
                        Q[j, j] += Q[i, j] * 2

            Q = torch.cat((Q[:i, :], Q[i + 1 :, :]), dim=0)
            Q = torch.cat((Q[:, :i], Q[:, i + 1 :]), dim=1)

        self.reduced_qubo.coefficients = Q
        self.reduced_qubo.update_metrics()

    def apply_rule(self, fixation_rule: Callable[[QUBOInstance], Dict[int, int]]) -> int:
        """
        Applies a given variable fixation rule to the reduced QUBO instance.

        Args:
            fixation_rule (Callable[[QUBOInstance], Dict[int, int]]): A function that
            returns a dictionary mapping variable indices to fixed values.

        Returns:
            int: The number of variables fixed by this rule.
        """
        fixed = fixation_rule(self.reduced_qubo)
        self.reduce_qubo(fixed)

        if fixed:
            self.fixed_var_dict_list.append(fixed)

        return len(fixed)

    def apply_full_fixation_exhaust(self) -> None:
        """
        Iteratively applies all fixation rules until no more variables can be fixed.

        This function repeatedly applies all rules in `self.fixation_rule_list`
        until no further reduction is possible.
        """
        fixed_sum = 1
        while fixed_sum > 0:
            fixed_sum = 0
            for fixation_rule in self.fixation_rule_list:
                fixed_var_number = self.apply_rule(fixation_rule)
                fixed_sum += fixed_var_number

    def post_process_fixation(self, solution: QUBOSolution) -> QUBOSolution:
        """
        Restores fixed variables in the solution bitstrings after QUBO reduction.

        This method reconstructs the full-length bitstrings by reinserting the fixed
        variables at their original positions.

        Args:
            solution (QUBOSolution): The solution object from the reduced QUBO problem.

        Returns:
            QUBOSolution: A solution object with bitstrings restored to their original size.
        """
        if not getattr(self.config, "do_preprocessing", False):
            return solution

        bitstring_list = solution.bitstrings.tolist() or [[]]

        def reinsert_fixed_variables(bitstring: List[int]) -> List[int]:
            for fixation_dict in reversed(self.fixed_var_dict_list):
                for position, bit_value in sorted(fixation_dict.items()):
                    bitstring.insert(position, bit_value)
            return bitstring

        should_restore = not self.config.use_quantum or (
            self.instance.size is not None and len(bitstring_list[0]) < self.instance.size
        )

        if should_restore:
            bitstring_list = [reinsert_fixed_variables(bitstring) for bitstring in bitstring_list]

        bitstrings = torch.tensor(bitstring_list, dtype=torch.float32)

        costs = torch.tensor(
            [self.instance.evaluate_solution(sample.tolist()) for sample in bitstrings],
            dtype=torch.float32,
        )

        return QUBOSolution(
            bitstrings=bitstrings,
            costs=costs,
            counts=solution.counts,
            probabilities=solution.probabilities,
            solution_status=SolutionStatusType.PREPROCESSED,
        )
