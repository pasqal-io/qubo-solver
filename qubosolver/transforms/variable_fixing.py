from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast, TypeAlias

import copy
import json
import torch

from qubosolver.types import QUBOSolution, bitstrings, vector
from qubosolver.types import QUBOInstance as QUBOInstanceBase
from qubosolver._io import utils as io_utils

#  TODO: Using `type` statement when Python >= 3.12
Rule: TypeAlias = Callable[[QUBOInstanceBase], dict[int, int]]


def hansen_fixing(qubo: QUBOInstanceBase) -> dict[int, int]:
    """
    Identifies and fixes variables in a QUBO instance based on threshold conditions.

    This method determines whether a variable should be fixed to 0 or 1 by computing
    lower and upper bounds from the diagonal and off-diagonal elements of the QUBO matrix.

    Args:
        qubo (QUBOInstance): The QUBO instance containing the coefficients matrix.

    Returns:
        dict[int, int]: A dictionary mapping variable indices to fixed values (0 or 1).
    """
    if qubo.matrix is None:
        raise ValueError("QUBO coefficients are not initialized.")

    fixed_dict: dict[int, int] = {}
    size_raw = qubo.size
    size: int = cast(int, size_raw)
    epsilon: float = 1e-8  # Tolerance to avoid floating-point precision issues

    for i in range(size):
        ci = qubo.matrix[i, i].item()  # Diagonal element

        q_minus = sum(min(0, qubo.matrix[i, j].item()) for j in range(size) if j != i)
        q_plus = sum(max(0, qubo.matrix[i, j].item()) for j in range(size) if j != i)

        if ci + q_minus * 2 >= -epsilon:
            fixed_dict[i] = 0
        elif ci + q_plus * 2 <= epsilon:
            fixed_dict[i] = 1

    return fixed_dict


class QUBOInstance(QUBOInstanceBase):
    """A QUBO instance with variable-fixing history.

    Wraps a parent :class:`~qubosolver.types.instance.QUBOInstance` and
    tracks which variables were fixed (and to which value) so the original
    solution can be reconstructed via :func:`unapply`.
    """

    def __init__(self, parent_instance: QUBOInstanceBase):
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original (unreduced) QUBO instance.
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(
            parent_instance.matrix,
        )
        self._parent_instance = copy.deepcopy(parent_instance)
        self._fixed_indices: list[dict[int, int]] = []

    @property
    def fixed_indices(self) -> list[dict[int, int]]:
        """Returns the history of variable-fixing.

        Returns:
            list[dict[int, int]]:
                List of dictionaries, where each dictionary represents a fixed variable
                and its value.
        """
        return self._fixed_indices

    @property
    def n_fixed_indices(self) -> int:
        """Returns the number of fixed variables.

        Returns:
            int: The number of fixed variables.
        """
        return sum([len(fixed) for fixed in self.fixed_indices])

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], instance: QUBOInstanceBase) -> None:
        """
        Saves a QUBOInstance to a file-like object.

        Args:
            file_like (io_utils.FileLike[bytes]):
                File-like object opened in binary write mode where the instance will be saved.
            instance (QUBOInstance):
                The QUBOInstance object to be saved.

        Returns:
            None
        """
        _check_QUBOInstance(instance)
        assert isinstance(instance, QUBOInstance)

        with io_utils.open(file_like, "wb") as f:
            QUBOInstanceBase.save(f, instance)
            QUBOInstanceBase.save(f, instance._parent_instance)
            fixed_var_json = json.dumps(instance._fixed_indices)
            io_utils.save_string(f, fixed_var_json)

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> QUBOInstance:
        """
        Loads a QUBOInstance from a file-like object.

        Args:
            file_like (io_utils.FileLike[bytes]):
                File-like object opened in binary read mode containing the saved QUBOInstance data.

        Returns:
            QUBOInstance:
                A new QUBOInstance object reconstructed from the saved data.
        """

        def decode_int_keys(obj: dict) -> dict:
            return {int(k): v for k, v in obj.items()}

        with io_utils.open(file_like, "rb") as f:
            instance = QUBOInstance(QUBOInstanceBase.load(f))
            instance._parent_instance = QUBOInstanceBase.load(f)
            fixed_var_json = io_utils.load_string(f)
            instance._fixed_indices = json.loads(fixed_var_json, object_hook=decode_int_keys)

        return instance


def _check_QUBOInstance(qubo: QUBOInstanceBase) -> None:
    """Raise :class:`TypeError` if *qubo* is not a variable-fixing :class:`QUBOInstance`."""
    if not isinstance(qubo, QUBOInstance):
        raise TypeError("Input must be an instance of _QUBOInstance.")


def _default_rules() -> tuple[Rule]:
    """Returns the default tuple of fixation rules (Hansen fixing)."""
    return (hansen_fixing,)


def _reduce_qubo(
    qubo: QUBOInstanceBase, fixed_indices: dict[int, int], *, inplace: bool = False
) -> QUBOInstance:
    """
    Applies variable fixation to reduce the size of the QUBO problem.

    This function modifies the QUBO coefficient matrix by:
    - Removing rows and columns corresponding to fixed variables.
    - Adjusting diagonal elements to account for fixed variables.

    Args:
        fixed_dict (dict[int, int]): A dictionary of fixed variable assignments.
            - Keys are variable indices.
            - Values are fixed binary values (0 or 1).

    Returns:
        None: Modifies `self.reduced_qubo` in place.
    """
    if not inplace:
        qubo = QUBOInstance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, QUBOInstance)

    if not fixed_indices:
        return qubo

    Q = qubo.matrix.clone()

    fixed_to_0 = {i for i, v in fixed_indices.items() if v == 0}
    fixed_to_1 = {i for i, v in fixed_indices.items() if v == 1}
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

    qubo._matrix = Q

    qubo._fixed_indices.append(fixed_indices)
    return qubo


def apply(
    qubo: QUBOInstanceBase,
    fixation_rules: Sequence[Rule] = _default_rules(),
    *,
    inplace: bool = False,
) -> QUBOInstance:
    """
    Applies a sequence of variable fixation rules to the QUBO instance.

    Args:
        qubo (QUBOInstance): The QUBO instance to apply rules to.
        fixation_rules (Sequence[FixationRule]): A sequence of functions that
            return dictionaries mapping variable indices to fixed values.

    Returns:
        list[dict[int, int]]: A list of fixation dictionaries, one per rule that fixed variables.
    """
    if not inplace:
        qubo = QUBOInstance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, QUBOInstance)

    for rule in fixation_rules:
        fixed = rule(qubo)
        _reduce_qubo(qubo, fixed, inplace=True)

    return qubo


def apply_recursively(
    qubo: QUBOInstanceBase,
    fixation_rules: Sequence[Rule] = _default_rules(),
    *,
    inplace: bool = False,
) -> QUBOInstance:
    """
    Iteratively applies all fixation rules until no more variables can be fixed.

    This function repeatedly applies all rules in `self.fixation_rule_list`
    until no further reduction is possible.
    """
    if not inplace:
        qubo = QUBOInstance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, QUBOInstance)

    while True:
        prev_n_fixations = len(qubo._fixed_indices)
        apply(qubo, fixation_rules, inplace=True)
        n_fixations = len(qubo._fixed_indices)
        assert n_fixations >= prev_n_fixations
        if n_fixations == prev_n_fixations:
            return qubo


def unapply(reduced_solution: QUBOSolution, reduced_qubo: QUBOInstance) -> QUBOSolution:
    """
    Restores fixed variables in the solution bitstrings after QUBO reduction.

    This method reconstructs the full-length bitstrings by reinserting the fixed
    variables at their original positions.

    Args:
        solution (QUBOSolution): The solution object from the reduced QUBO problem.

    Returns:
        QUBOSolution: A solution object with bitstrings restored to their original size.
    """
    # FIXME: raise if empty solution ?
    bitstrings_list = reduced_solution.bitstrings.tolist() or [[]]

    def reinsert_fixed_variables(bitstring: list[int]) -> list[int]:
        for fixation_dict in reversed(reduced_qubo._fixed_indices):
            for position, bit_value in sorted(fixation_dict.items()):
                bitstring.insert(position, bit_value)
        return bitstring

    bits_to_reinsert = sum(len(fixation_dict) for fixation_dict in reduced_qubo._fixed_indices)
    assert (bits_to_reinsert + len(bitstrings_list[0])) == reduced_qubo._parent_instance.size

    if bits_to_reinsert == 0:
        return copy.deepcopy(reduced_solution)

    solution = QUBOSolution()

    solution.bitstrings = bitstrings.tensor(
        [reinsert_fixed_variables(bitstring) for bitstring in bitstrings_list]
    )
    solution.costs = vector.tensor(
        [reduced_qubo._parent_instance.evaluate_solution(b) for b in solution.bitstrings]
    )
    solution.counts = reduced_solution.counts
    solution.probabilities = reduced_solution.probabilities

    return solution
