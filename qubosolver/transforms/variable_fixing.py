"""Variable-fixing transforms for QUBO problem reduction.

Variable fixing eliminates variables from a QUBO instance before solving by
proving, from the structure of the objective matrix alone, that certain
variables must be 0 or 1 in any optimal solution.  Reducing the problem size
this way can significantly cut the resources required by the solver.

Typical usage:

```python
from qubosolver.transforms import variable_fixing
from qubosolver.solving import brute_force

reduced_instance = variable_fixing.apply_recursively(instance)
reduced_solution = brute_force.solve(reduced_instance)
solution = variable_fixing.lift(reduced_solution, reduced_instance)
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

# TODO: Using `type` statement when Python >= 3.12
Rule: TypeAlias = Callable[[qubosolver.Instance], dict[int, int]]
"""A function that inspects a QUBO instance and returns the variables it can fix,
as a mapping from variable index to the value (0 or 1) it is fixed to."""


def hansen_fixing(instance: qubosolver.Instance) -> dict[int, int]:
    """Identify variables that can be fixed using Hansen's bounding criterion.

    For each variable *i*, computes a lower bound
    ``c_i + 2 * sum(min(0, Q_ij))`` and an upper bound
    ``c_i + 2 * sum(max(0, Q_ij))`` from the diagonal and off-diagonal
    elements of the QUBO matrix.  A variable is fixed to 0 when its lower
    bound is non-negative (it cannot improve the objective by being 1) and
    to 1 when its upper bound is non-positive (it can only improve it).

    Args:
        instance: The QUBO instance to analyze.

    Returns:
        Mapping of variable index to fixed value (``0`` or ``1``).
            Variables that cannot be fixed are omitted.
    """
    fixed_dict: dict[int, int] = {}
    size: int = cast(int, instance.size)
    epsilon: float = 1e-8  # Tolerance to avoid floating-point precision issues

    for i in range(size):
        ci = instance.matrix[i, i].item()  # Diagonal element

        q_minus = sum(min(0, instance.matrix[i, j].item()) for j in range(size) if j != i)
        q_plus = sum(max(0, instance.matrix[i, j].item()) for j in range(size) if j != i)

        if ci + q_minus * 2 >= -epsilon:
            fixed_dict[i] = 0
        elif ci + q_plus * 2 <= epsilon:
            fixed_dict[i] = 1

    return fixed_dict


class Instance(qubosolver.Instance):
    """A QUBO instance with variable-fixing history.

    Wraps a parent [`qubosolver.Instance`][] and
    tracks which variables were fixed (and to which value) so the original
    solution can be reconstructed via [`lift`][].
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
        """Fixation history: one dict per [`apply`][] call, mapping index → fixed value."""
        return self._fixed_indices

    @property
    def n_fixed_indices(self) -> int:
        """Total number of variables fixed across all fixation rounds."""
        return sum([len(fixed) for fixed in self.fixed_indices])

    def _write_body(self, f: io_utils.FileLike[bytes]) -> None:
        """Write the parent matrix, fixation history, and parent instance to `f`."""
        super()._write_body(f)
        io_utils.save_string(f, json.dumps(self._fixed_indices))  # type: ignore[arg-type]
        self._parent_instance.save(f)  # type: ignore[arg-type]

    @classmethod
    def _read_body(cls, f: io_utils.FileLike[bytes]) -> Instance:
        """Read back a variable-fixing instance written by [`_write_body`][]."""

        def decode_int_keys(obj: dict) -> dict:
            return {int(k): v for k, v in obj.items()}

        instance = Instance(qubosolver.Instance._read_body(f))
        instance._fixed_indices = json.loads(
            io_utils.load_string(f), object_hook=decode_int_keys  # type: ignore[arg-type]
        )
        instance._parent_instance = qubosolver.Instance.load(f)  # type: ignore[arg-type]
        return instance


def _check_QUBOInstance(qubo: qubosolver.Instance) -> None:
    """Raise `TypeError` if *qubo* is not a variable-fixing `Instance`."""
    if not isinstance(qubo, Instance):
        raise TypeError("Input must be an instance of _QUBOInstance.")


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
    instance: qubosolver.Instance,
    fixation_rules: Sequence[Rule] = (hansen_fixing,),
    *,
    inplace: bool = False,
) -> Instance:
    """Apply each fixation rule once and reduce the QUBO matrix accordingly.

    Each rule in `fixation_rules` is called in order; variables it identifies
    are immediately fixed and the matrix is reduced before the next rule runs.

    Args:
        instance: The QUBO instance to reduce.
        fixation_rules: Ordered sequence of [`Rule`][] callables.
        inplace: If ``False`` (default), wraps `instance` in a new
            variable-fixing [`Instance`][] before modifying it.

    Returns:
        The reduced instance with updated fixation history.
    """
    if not inplace:
        instance = Instance(instance)

    _check_QUBOInstance(instance)
    assert isinstance(instance, Instance)  # nosec B101

    for rule in fixation_rules:
        fixed = rule(instance)
        _reduce_qubo(instance, fixed, inplace=True)

    return instance


def apply_recursively(
    instance: qubosolver.Instance,
    fixation_rules: Sequence[Rule] = (hansen_fixing,),
    *,
    inplace: bool = False,
) -> Instance:
    """Apply fixation rules repeatedly until no further variables can be fixed.

    Calls [`apply`][] in a loop; stops when a full pass over all rules
    fixes no additional variables.

    Args:
        instance: The QUBO instance to reduce.
        fixation_rules: Ordered sequence of [`Rule`] callables.
        inplace: If ``False`` (default), wraps `instance` in a new
            variable-fixing [`Instance`][] before modifying it.

    Returns:
        The fully reduced instance.
    """
    if not inplace:
        instance = Instance(instance)

    _check_QUBOInstance(instance)
    assert isinstance(instance, Instance)  # nosec B101

    while True:
        prev_n_fixations = len(instance._fixed_indices)
        apply(instance, fixation_rules, inplace=True)
        n_fixations = len(instance._fixed_indices)
        assert n_fixations >= prev_n_fixations  # nosec B101
        if n_fixations == prev_n_fixations:
            return instance


def lift(reduced_solution: Solution, reduced_instance: Instance) -> Solution:
    """Reconstruct the full solution by reinserting fixed variables.

    Reverses the fixation history stored in `reduced_instance`: fixed variables
    are reinserted at their original positions in each bitstring, and costs
    are recomputed against the original (unreduced) QUBO matrix.

    If no variables were fixed, returns a deep copy of `reduced_solution`
    unchanged.

    Args:
        reduced_solution: Solution obtained from solving the reduced QUBO.
        reduced_instance: The reduced instance carrying the fixation
            history and a reference to the original instance.

    Returns:
        A new solution with full-length
            bitstrings and costs evaluated against the original QUBO matrix.
            Counts and probabilities are carried over from `reduced_solution`.
    """
    bitstrings_list = reduced_solution.bitstrings.tolist() or [[]]

    def reinsert_fixed_variables(bitstring: list[int]) -> list[int]:
        for fixation_dict in reversed(reduced_instance._fixed_indices):
            for position, bit_value in sorted(fixation_dict.items()):
                bitstring.insert(position, bit_value)
        return bitstring

    bits_to_reinsert = sum(len(fixation_dict) for fixation_dict in reduced_instance._fixed_indices)
    assert (
        bits_to_reinsert + len(bitstrings_list[0])
    ) == reduced_instance._parent_instance.size  # nosec B101

    if bits_to_reinsert == 0:
        return copy.deepcopy(reduced_solution)

    solution = Solution()

    solution.bitstrings = bitstrings.tensor(
        [reinsert_fixed_variables(bitstring) for bitstring in bitstrings_list]
    )
    solution.costs = vector.tensor(
        [reduced_instance._parent_instance.cost(b) for b in solution.bitstrings]
    )
    solution.counts = reduced_solution.counts
    solution.probabilities = reduced_solution.probabilities

    return solution
