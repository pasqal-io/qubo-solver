# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

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

import copy
import json
import logging
from collections.abc import Callable, Sequence

import qubosolver
from qubosolver._checks import TYPE_CHECKING
from qubosolver._io import utils as io_utils
from qubosolver.types import Solution, bitstrings, vectori

if TYPE_CHECKING:
    from qubosolver._io.utils import Stream
    from qubosolver.types import Candidate

logger = logging.getLogger(__name__)

# TODO(abussy-pasqal): Using `type` statement when Python >= 3.12
Rule = Callable[[qubosolver.Instance], dict[int, int]]
"""A function that inspects a QUBO instance and returns the variables it can fix,
as a mapping from variable index, in ``range(instance.size)``, to the value (0 or 1)
it is fixed to.

To restrict what a rule fixes, wrap it, e.g. to only fix variables to 0:
``lambda instance: {i: v for i, v in hansen_fixing(instance).items() if v == 0}``."""


def hansen_fixing(instance: qubosolver.Instance) -> dict[int, int]:
    """Identify variables that can be fixed using Hansen's bounding criterion.

    For each variable *i*, computes a lower bound
    ``c_i + 2 * sum(min(0, Q_ij))`` and an upper bound
    ``c_i + 2 * sum(max(0, Q_ij))`` from the diagonal and off-diagonal
    elements of the QUBO matrix.  A variable is fixed to 0 when its lower
    bound is positive (it cannot improve the objective by being 1) and
    to 1 when its upper bound is negative (it can only improve it), both
    by a margin of ``1e-8`` against rounding errors.

    Args:
        instance: The QUBO instance to analyze.

    Returns:
        Mapping of variable index to fixed value (``0`` or ``1``).
            Variables that cannot be fixed are omitted.
    """
    # TODO(abussy-pasqal): The tolerance is absolute: it refuses exact ties (a bound
    # of exactly 0, e.g. a zero row) and is meaningless on a matrix scaled far
    # from 1. Make it either 0 (exact criterion) or atol + rtol * the magnitude of
    # the summed terms (|c_i| + 2 * sum(|Q_ij|) over the couplings summed in that
    # bound).
    epsilon = 1e-8  # Conservative margin against floating-point rounding

    diagonal = instance.matrix.diagonal()
    off_diagonal = instance.matrix.clone().fill_diagonal_(0)
    lower = diagonal + 2 * off_diagonal.clamp(max=0).sum(dim=1)
    upper = diagonal + 2 * off_diagonal.clamp(min=0).sum(dim=1)

    to_0 = lower >= epsilon
    to_1 = upper <= -epsilon
    fixed = (to_0 | to_1).nonzero().flatten()
    # to_1 is False both for variables fixed to 0 and for unfixed ones; restricted
    # to the fixed variables, False only leaves those of to_0.
    return dict(zip(fixed.tolist(), to_1[fixed].int().tolist(), strict=True))


class Instance(qubosolver.Instance):
    """A QUBO instance with variable-fixing history.

    Wraps a parent [`qubosolver.Instance`][] and
    tracks which variables were fixed (and to which value) so the original
    solution can be reconstructed via [`lift`][].
    """

    def __init__(self, parent_instance: qubosolver.Instance) -> None:
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original (unreduced) QUBO instance.
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(
            parent_instance.matrix.detach().clone(),
        )
        self._parent_instance = copy.deepcopy(parent_instance)
        self._fixed_history: list[dict[int, int]] = []

    @property
    def fixed_history(self) -> list[dict[int, int]]:
        """Fixation history: one dict per rule application that fixed variables.

        Each dict maps original index → fixed value, in the order the rules were
        applied: [`apply`][] adds one dict per rule that fixes at least one
        variable. Indices refer to the variables of the parent instance, not of
        the reduced one, and each dict is sorted by index.

        A fresh copy on each access: mutating it does not change the history.
        """
        return copy.deepcopy(self._fixed_history)

    @property
    def fixed_variables(self) -> dict[int, int]:
        """All fixed variables, merged over [`fixed_history`][], as original index → fixed value.

        A fresh dict on each access: mutating it does not change the history.
        """
        return {i: v for fixed in self._fixed_history for i, v in fixed.items()}

    def _remaining_variables(self) -> list[int]:
        """Original indices of the variables not fixed yet, in increasing order.

        Reduction never reorders variables, so reduced variable *i* is the *i*-th entry.
        """
        fixed = self.fixed_variables
        return [i for i in range(self._parent_instance.size) if i not in fixed]

    def _write_body(self, f: Stream[bytes]) -> None:
        """Write the parent matrix, fixation history, and parent instance to `f`."""
        super()._write_body(f)
        io_utils.save_string(f, json.dumps(self._fixed_history))
        self._parent_instance.save(f)

    @classmethod
    def _read_body(cls, f: Stream[bytes]) -> Instance:
        """Read back a variable-fixing instance written by [`_write_body`][]."""

        def decode_int_keys(obj: dict) -> dict:
            return {int(k): v for k, v in obj.items()}

        instance = Instance(qubosolver.Instance._read_body(f))
        instance._fixed_history = json.loads(io_utils.load_string(f), object_hook=decode_int_keys)
        instance._parent_instance = qubosolver.Instance.load(f)
        return instance


def _reduce_qubo(
    instance: qubosolver.Instance,
    fixed_indices: dict[int, int],
    *,
    inplace: bool = False,
) -> Instance:
    """Reduce the QUBO matrix by fixing a set of variables.

    For each variable fixed to 1, its interaction terms are folded into the
    diagonal of the remaining variables before its row and column are removed.
    Variables fixed to 0 are removed without any adjustment.

    Args:
        instance: The QUBO instance to reduce.
        fixed_indices: Mapping of variable index to fixed value (``0`` or ``1``).
        inplace: If ``False`` (default), wraps *qubo* in a new
            `Instance` before modifying it.

    Returns:
        The (possibly new) instance with the reduced matrix and
        *fixed_indices*, translated to original indices, appended to its
        fixation history.

    Raises:
        ValueError: If an index of *fixed_indices* is outside ``range(instance.size)``.
    """
    if out_of_range := sorted(i for i in fixed_indices if not 0 <= i < instance.size):
        msg = f"Fixed indices {out_of_range} are outside range({instance.size}) of the instance."
        raise ValueError(msg)

    if not inplace:
        instance = Instance(instance)

    instance = instance.variable_fixing

    if not fixed_indices:
        return instance

    remaining = instance._remaining_variables()
    kept = vectori.tensor([i for i in range(instance.size) if i not in fixed_indices])
    fixed_to_1 = vectori.tensor([i for i, v in fixed_indices.items() if v == 1])

    rows = instance.matrix.index_select(0, kept)
    reduced = rows.index_select(1, kept)
    reduced.diagonal().add_(2 * rows.index_select(1, fixed_to_1).sum(dim=1))
    instance._matrix = reduced

    instance._fixed_history.append(
        dict(sorted((remaining[i], v) for i, v in fixed_indices.items()))
    )
    return instance


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

    Raises:
        ValueError: If a rule returns an index outside ``range(instance.size)``.
    """
    if not inplace:
        instance = Instance(instance)

    instance = instance.variable_fixing

    for rule in fixation_rules:
        fixed = rule(instance)
        _reduce_qubo(instance, fixed, inplace=True)

    if instance.size == 0:
        logger.info("Variable fixing reduced the instance to zero variables.")

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

    Raises:
        ValueError: If a rule returns an index outside ``range(instance.size)``.
    """
    if not inplace:
        instance = Instance(instance)

    instance = instance.variable_fixing

    while True:
        prev_n_fixations = len(instance._fixed_history)
        apply(instance, fixation_rules, inplace=True)
        n_fixations = len(instance._fixed_history)
        assert n_fixations >= prev_n_fixations  # nosec B101
        if n_fixations == prev_n_fixations:
            return instance


def lift(reduced_solution: Solution, reduced_instance: Instance) -> Solution:
    """Reconstruct the full solution by reinserting fixed variables.

    Writes the fixed variables of `reduced_instance` at their original
    positions in each bitstring, fills the remaining positions, in order,
    with the reduced bitstring, and recomputes costs against the original
    (unreduced) QUBO matrix.

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
    if not reduced_solution:
        return Solution()

    fixed = reduced_instance.fixed_variables
    parent_size = reduced_instance._parent_instance.size
    assert len(fixed) + reduced_solution.num_variables == parent_size  # nosec B101

    if not fixed:
        return copy.deepcopy(reduced_solution)

    full = bitstrings.zeros(len(reduced_solution), parent_size)
    full[:, list(fixed)] = bitstrings.tensor([list(fixed.values())])
    full[:, reduced_instance._remaining_variables()] = reduced_solution.bitstrings

    return Solution(
        bitstrings=full,
        counts=reduced_solution.counts,
        probabilities=reduced_solution.probabilities,
    )._compute_costs(reduced_instance._parent_instance.matrix)


def from_candidate(
    candidate: Candidate, variables: Sequence[int], instance: qubosolver.Instance
) -> Rule:
    """Turn a candidate of a sub-instance into a fixation rule on `instance`.

    The candidate's bit *i* is the value of variable `variables[i]` of `instance`,
    as for a sub-instance built with [`qubosolver.Instance.subinstance`][]. The
    returned rule fixes those variables to those values.

    Choosing which candidate to fix is the caller's decision: `solution[0]` is the
    best one of a solution sorted by cost (e.g. from `Solution.from_results`).

    The rule is only valid on an instance of the same size as `instance`, so it
    can only be applied once, with [`apply`][], and must come first in
    `fixation_rules`: a rule running before it in the same call may fix variables
    and shift the indices. Do not pass it to [`apply_recursively`][]: once its
    variables are fixed, the instance is smaller and the next pass raises. Calling
    the rule directly, `rule(instance)`, returns the fixation dict.

    Args:
        candidate: The candidate whose bits are fixed.
        variables: Distinct variable indices of `instance`, in the order of the
            candidate's bits.
        instance: The instance the indices in `variables` refer to.

    Returns:
        A [`Rule`][] fixing `variables` to the candidate's bits.

    Raises:
        ValueError: If the candidate's width differs from `len(variables)`, or
            if `variables` contains duplicates or out-of-range indices. The rule
            itself raises `ValueError` when applied to an instance whose size
            differs from `instance.size`.
    """
    size = instance.size
    if len(candidate.bitstring) != len(variables):
        msg = (
            f"Candidate has {len(candidate.bitstring)} bits but {len(variables)} variables "
            "were given."
        )
        raise ValueError(msg)
    if len(set(variables)) != len(variables):
        msg = f"Variables must be distinct, got {variables}."
        raise ValueError(msg)
    if any(not 0 <= v < size for v in variables):
        msg = f"Variables must be in range({size}), got {variables}."
        raise ValueError(msg)

    fixation = dict(zip(variables, candidate.bitstring.tolist(), strict=True))

    def rule(target: qubosolver.Instance) -> dict[int, int]:
        if target.size != size:
            msg = (
                f"This rule was built for an instance of size {size}, but was applied to one "
                f"of size {target.size}: apply it once, with `apply`, first in `fixation_rules`."
            )
            raise ValueError(msg)
        return fixation.copy()

    return rule
