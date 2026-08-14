"""Variable-fixing transforms for QUBO problem reduction.

Variable fixing eliminates variables from a QUBO instance before solving by
proving, from the structure of the objective matrix alone, that certain
variables must be 0 or 1 in any optimal solution.  Reducing the problem size
this way can significantly cut the resources required by the solver.

Typical usage:

```python
import qubosolver.transforms.variable_fixing as vf

reduced_instance = vf.apply_recursively(qubo_instance)
reduced_solution = solver.solve(reduced_instance)
full_solution = vf.lift(reduced_solution, reduced)
```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast, TypeAlias

import copy
import json
import torch

import qubosolver
from qubosolver.types import Solution, bitstrings, vector
from qubosolver._io import utils as io_utils

#  TODO: Using `type` statement when Python >= 3.12
Rule: TypeAlias = Callable[[qubosolver.Instance], dict[int, int]]


def hansen_fixing(qubo: qubosolver.Instance) -> dict[int, int]:
    """Identify variables that can be fixed using Hansen's bounding criterion.

    For each variable *i*, computes a lower bound
    ``c_i + 2 * sum(min(0, Q_ij))`` and an upper bound
    ``c_i + 2 * sum(max(0, Q_ij))`` from the diagonal and off-diagonal
    elements of the QUBO matrix.  A variable is fixed to 0 when its lower
    bound is non-negative (it cannot improve the objective by being 1) and
    to 1 when its upper bound is non-positive (it can only improve it).

    Args:
        qubo: The QUBO instance to analyse.

    Returns:
        Mapping of variable index to fixed value (``0`` or ``1``).
        Variables that cannot be fixed are omitted.

    Raises:
        ValueError: If the QUBO matrix is not initialised.
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


class Instance(qubosolver.Instance):
    """A QUBO instance with variable-fixing history.

    Wraps a parent [`qubosolver.Instance`][] and
    tracks which variables were fixed (and to which value) so the original
    solution can be reconstructed via `lift`.
    """

    def __init__(self, parent_instance: qubosolver.Instance):
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original (unreduced) QUBO instance.
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(
            parent_instance.matrix.detach().clone(),
        )
        self._parent_instance = copy.deepcopy(parent_instance)
        self._fixed_indices: list[dict[int, int]] = []

    @property
    def fixed_indices(self) -> list[dict[int, int]]:
        """Fixation history: one dict per `apply` call, mapping index → fixed value."""
        return self._fixed_indices

    @property
    def n_fixed_indices(self) -> int:
        """Total number of variables fixed across all fixation rounds."""
        return sum([len(fixed) for fixed in self.fixed_indices])

    @staticmethod
    def save(file_like: io_utils.FileLike[bytes], instance: qubosolver.Instance) -> None:
        """Serialise an `Instance` (including fixation history) to *file_like*.

        Args:
            file_like: Binary-writable file-like object or path.
            instance: The instance to save.

        Raises:
            TypeError: If *instance* is not a variable-fixing `Instance`.
        """
        _check_QUBOInstance(instance)
        assert isinstance(instance, Instance)  # nosec B101

        with io_utils.open(file_like, "wb") as f:
            qubosolver.Instance.save(f, instance)
            qubosolver.Instance.save(f, instance._parent_instance)
            fixed_var_json = json.dumps(instance._fixed_indices)
            io_utils.save_string(f, fixed_var_json)

    @staticmethod
    def load(file_like: io_utils.FileLike[bytes]) -> Instance:
        """Deserialise an `Instance` (including fixation history) from *file_like*.

        Args:
            file_like: Binary-readable file-like object or path produced by `save`.

        Returns:
            The reconstructed instance.
        """

        def decode_int_keys(obj: dict) -> dict:
            return {int(k): v for k, v in obj.items()}

        with io_utils.open(file_like, "rb") as f:
            instance = Instance(qubosolver.Instance.load(f))
            instance._parent_instance = qubosolver.Instance.load(f)
            fixed_var_json = io_utils.load_string(f)
            instance._fixed_indices = json.loads(fixed_var_json, object_hook=decode_int_keys)

        return instance


def _check_QUBOInstance(qubo: qubosolver.Instance) -> None:
    """Raise `TypeError` if *qubo* is not a variable-fixing `Instance`."""
    if not isinstance(qubo, Instance):
        raise TypeError("Input must be an instance of _QUBOInstance.")


def _default_rules() -> tuple[Rule]:
    """Returns the default tuple of fixation rules (Hansen fixing)."""
    return (hansen_fixing,)


def _reduce_qubo(
    qubo: qubosolver.Instance, fixed_indices: dict[int, int], *, inplace: bool = False
) -> Instance:
    """Reduce the QUBO matrix by fixing a set of variables.

    For each variable fixed to 1, its interaction terms are folded into the
    diagonal of the remaining variables before its row and column are removed.
    Variables fixed to 0 are removed without any adjustment.

    Args:
        qubo: The QUBO instance to reduce.
        fixed_indices: Mapping of variable index to fixed value (``0`` or ``1``).
        inplace: If ``False`` (default), wraps *qubo* in a new
            `Instance` before modifying it.

    Returns:
        The (possibly new) instance with the reduced matrix and
        *fixed_indices* appended to its fixation history.
    """
    if not inplace:
        qubo = Instance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, Instance)  # nosec B101

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
    qubo: qubosolver.Instance,
    fixation_rules: Sequence[Rule] = _default_rules(),
    *,
    inplace: bool = False,
) -> Instance:
    """Apply each fixation rule once and reduce the QUBO matrix accordingly.

    Each rule in *fixation_rules* is called in order; variables it identifies
    are immediately fixed and the matrix is reduced before the next rule runs.

    Args:
        qubo: The QUBO instance to reduce.
        fixation_rules: Ordered sequence of `Rule` callables.
            Defaults to ``(hansen_fixing,)``.
        inplace: If ``False`` (default), wraps *qubo* in a new
            `Instance` before modifying it.

    Returns:
        The reduced instance with updated fixation history.
    """
    if not inplace:
        qubo = Instance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, Instance)  # nosec B101

    for rule in fixation_rules:
        fixed = rule(qubo)
        _reduce_qubo(qubo, fixed, inplace=True)

    return qubo


def apply_recursively(
    qubo: qubosolver.Instance,
    fixation_rules: Sequence[Rule] = _default_rules(),
    *,
    inplace: bool = False,
) -> Instance:
    """Apply fixation rules repeatedly until no further variables can be fixed.

    Calls `apply` in a loop; stops when a full pass over all rules
    fixes no additional variables.

    Args:
        qubo: The QUBO instance to reduce.
        fixation_rules: Ordered sequence of `Rule` callables.
            Defaults to ``(hansen_fixing,)``.
        inplace: If ``False`` (default), wraps *qubo* in a new
            `Instance` before modifying it.

    Returns:
        The fully reduced instance.
    """
    if not inplace:
        qubo = Instance(qubo)

    _check_QUBOInstance(qubo)
    assert isinstance(qubo, Instance)  # nosec B101

    while True:
        prev_n_fixations = len(qubo._fixed_indices)
        apply(qubo, fixation_rules, inplace=True)
        n_fixations = len(qubo._fixed_indices)
        assert n_fixations >= prev_n_fixations  # nosec B101
        if n_fixations == prev_n_fixations:
            return qubo


def lift(reduced_solution: Solution, reduced_qubo: Instance) -> Solution:
    """Reconstruct the full solution by reinserting fixed variables.

    Reverses the fixation history stored in *reduced_qubo*: fixed variables
    are reinserted at their original positions in each bitstring, and costs
    are recomputed against the original (unreduced) QUBO matrix.

    If no variables were fixed, returns a deep copy of *reduced_solution*
    unchanged.

    Args:
        reduced_solution: Solution obtained from solving the reduced QUBO.
        reduced_qubo: The reduced instance carrying the fixation
            history and a reference to the original instance.

    Returns:
        A new solution with full-length
            bitstrings and costs evaluated against the original QUBO matrix.
            Counts and probabilities are carried over from *reduced_solution*.
    """
    bitstrings_list = reduced_solution.bitstrings.tolist() or [[]]

    def reinsert_fixed_variables(bitstring: list[int]) -> list[int]:
        for fixation_dict in reversed(reduced_qubo._fixed_indices):
            for position, bit_value in sorted(fixation_dict.items()):
                bitstring.insert(position, bit_value)
        return bitstring

    bits_to_reinsert = sum(len(fixation_dict) for fixation_dict in reduced_qubo._fixed_indices)
    assert (
        bits_to_reinsert + len(bitstrings_list[0])
    ) == reduced_qubo._parent_instance.size  # nosec B101

    if bits_to_reinsert == 0:
        return copy.deepcopy(reduced_solution)

    solution = Solution()

    solution.bitstrings = bitstrings.tensor(
        [reinsert_fixed_variables(bitstring) for bitstring in bitstrings_list]
    )
    solution.costs = vector.tensor(
        [reduced_qubo._parent_instance.cost(b) for b in solution.bitstrings]
    )
    solution.counts = reduced_solution.counts
    solution.probabilities = reduced_solution.probabilities

    return solution
