"""Bit-flip preprocessing transforms for QUBO instances with negative interactions.

Quantum (Rydberg) solvers cannot encode attractive interactions, so a QUBO must
have non-negative off-diagonal coefficients to be embeddable.  A change of
variable ``x_i -> 1 - y_i`` on a subset of variables flips the sign of the
interactions incident to it; choosing the subset that removes as much negative
weight as possible is an integer linear program solved here with GLPK.

[`apply`][qubosolver.transforms.negative_bitflip.apply] solves the ILP and
applies the optimal bit flips to the matrix, returning a wrapper `Instance` that
records the flip vector so the solution can later be mapped back with
[`unapply`][qubosolver.transforms.negative_bitflip.unapply].  When bit flips
cannot remove *every* negative off-diagonal coefficient, the remaining ones can
be dropped with
[`qubosolver.transforms.zeroing.apply`][qubosolver.transforms.zeroing.apply].

Typical usage:

```python
import qubosolver.transforms.negative_bitflip as bitflip

reduced = bitflip.apply(qubo_instance, time_limit_s=60.0)
solution = solver.solve(reduced)
full = bitflip.unapply(solution, reduced)
```
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch

import qubosolver
from qubosolver.types import Solution, vector, Matrix, Bitstrings, Bitstring, bitstring

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Edge:
    """Non-zero upper-triangular QUBO interaction."""

    i: int
    j: int
    q: float


def _has_negative_offdiagonal(Q: Matrix, eps: float = 0.0) -> bool:
    """Return True if Q has at least one negative off-diagonal coefficient."""
    n = Q.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=Q.device)
    return bool(torch.any(Q[offdiag_mask] < -eps).item())


def _extract_upper_edges(Q: Matrix, eps: float = 0.0) -> list[_Edge]:
    """Extract non-zero upper-triangular off-diagonal QUBO coefficients."""
    n = Q.shape[0]
    edges: list[_Edge] = []

    for i in range(n):
        for j in range(i + 1, n):
            q = float(Q[i, j].item())
            if abs(q) > eps:
                edges.append(_Edge(i=i, j=j, q=q))

    return edges


def _apply_bitflips(
    bitstrings: Bitstrings,
    flips: Bitstring,
) -> Bitstrings:
    """Apply or undo bit flips on bitstrings.

    The operation is its own inverse.
    """
    if bitstrings.numel() == 0:
        return bitstrings
    return bitstrings ^ flips.unsqueeze(0)


def _transform_qubo_with_bitflips(
    Q: Matrix,
    flips: Bitstring,
) -> tuple[Matrix, float]:
    """Transform QUBO coefficients after variable bit flips.

    Convention:
        x_i = y_i       if flips_i = 0
        x_i = 1 - y_i   if flips_i = 1

    The returned offset satisfies:
        x^T Q x = y^T Q_flipped y + offset
    """

    f = flips.to(dtype=Q.dtype)
    s = 1.0 - 2.0 * f
    linear = 2.0 * s * (Q @ f)
    Q_flipped = Q * torch.outer(s, s) + torch.diag(linear)

    offset = float(f @ Q @ f)

    return Q_flipped, offset


def _coefficient_after_bitflip(q: float, fi: int, fj: int) -> float:
    """Return the off-diagonal coefficient sign after bit flips."""
    xor = fi ^ fj
    return (1.0 - 2.0 * xor) * q


def _compute_negative_weight_metrics(
    Q: Matrix,
    flips: Bitstring,
    eps: float = 0.0,
) -> dict[str, Any]:
    """Compute negative off-diagonal count and weight before and after bit flips."""
    edges = _extract_upper_edges(Q, eps=eps)

    neg_count_before = 0
    neg_count_after = 0
    neg_weight_before = 0.0
    neg_weight_after = 0.0

    for edge in edges:
        q = edge.q

        if q < -eps:
            neg_count_before += 1
            neg_weight_before += abs(q)

        q_after = _coefficient_after_bitflip(q, int(flips[edge.i]), int(flips[edge.j]))
        if q_after < -eps:
            neg_count_after += 1
            neg_weight_after += abs(q_after)

    def reduction_pct(before: float, after: float) -> float:
        if before <= 0:
            return 0.0
        return 100.0 * (before - after) / before

    return {
        "n_edges": len(edges),
        "neg_count_before": neg_count_before,
        "neg_count_after": neg_count_after,
        "neg_count_reduction_pct": reduction_pct(neg_count_before, neg_count_after),
        "neg_weight_before": neg_weight_before,
        "neg_weight_after": neg_weight_after,
        "neg_weight_reduction_pct": reduction_pct(neg_weight_before, neg_weight_after),
    }


def _solve_bitflip_preprocessing_glpk(
    Q: Matrix,
    *,
    time_limit_s: float = 60.0,
    eps: float = 0.0,
    log: bool = False,
) -> tuple[torch.Tensor, float, str]:
    """Solve the GLPK bit-flip preprocessing ILP with the negative-weight objective.

    Variables:
        f_i in {0, 1}
        y_k in {0, 1}, with y_k = f_i XOR f_j

    Objective:
        minimize the total absolute weight of negative off-diagonal coefficients
        remaining after applying the variable flips.
    """
    import swiglpk as glp

    n = Q.shape[0]
    edges = _extract_upper_edges(Q, eps=eps)
    m = len(edges)

    def f_col(i: int) -> int:
        return i + 1

    def y_col(k: int) -> int:
        return n + k + 1

    if m == 0:
        return bitstring.zeros(n), 0.0, "OPTIMAL"

    prob = glp.glp_create_prob()

    try:
        glp.glp_set_prob_name(prob, "bitflip_negative_weight_preprocessing")
        glp.glp_set_obj_dir(prob, glp.GLP_MIN)

        glp.glp_add_cols(prob, n + m)

        for i in range(n):
            col = f_col(i)
            glp.glp_set_col_name(prob, col, f"f_{i}")
            glp.glp_set_col_kind(prob, col, glp.GLP_BV)
            glp.glp_set_col_bnds(prob, col, glp.GLP_DB, 0.0, 1.0)
            glp.glp_set_obj_coef(prob, col, 0.0)

        for k, edge in enumerate(edges):
            col = y_col(k)
            glp.glp_set_col_name(prob, col, f"y_{k}")
            glp.glp_set_col_kind(prob, col, glp.GLP_BV)
            glp.glp_set_col_bnds(prob, col, glp.GLP_DB, 0.0, 1.0)

            weight = abs(float(edge.q))

            # Negative remaining weight:
            #   q < 0 : |q| * (1 - y) = constant - |q| * y
            #   q > 0 : |q| * y
            objective_coefficient = -weight if edge.q < 0 else weight
            glp.glp_set_obj_coef(prob, col, objective_coefficient)

        glp.glp_add_rows(prob, 4 * m)

        row = 1

        def set_three_term_row(
            row_index: int,
            bound_type: int,
            lower_bound: float,
            upper_bound: float,
            columns: tuple[int, int, int],
            coefficients: tuple[float, float, float],
        ) -> None:
            glp.glp_set_row_bnds(
                prob,
                row_index,
                bound_type,
                lower_bound,
                upper_bound,
            )

            indices = glp.intArray(4)
            values = glp.doubleArray(4)

            for position, (column, coefficient) in enumerate(
                zip(columns, coefficients),
                start=1,
            ):
                indices[position] = column
                values[position] = coefficient

            glp.glp_set_mat_row(prob, row_index, 3, indices, values)

        for k, edge in enumerate(edges):
            fi = f_col(edge.i)
            fj = f_col(edge.j)
            y = y_col(k)

            # y >= fi - fj  -> y - fi + fj >= 0
            set_three_term_row(
                row,
                glp.GLP_LO,
                0.0,
                0.0,
                (y, fi, fj),
                (1.0, -1.0, 1.0),
            )
            row += 1

            # y >= fj - fi  -> y + fi - fj >= 0
            set_three_term_row(
                row,
                glp.GLP_LO,
                0.0,
                0.0,
                (y, fi, fj),
                (1.0, 1.0, -1.0),
            )
            row += 1

            # y <= fi + fj  -> y - fi - fj <= 0
            set_three_term_row(
                row,
                glp.GLP_UP,
                0.0,
                0.0,
                (y, fi, fj),
                (1.0, -1.0, -1.0),
            )
            row += 1

            # y <= 2 - fi - fj  -> y + fi + fj <= 2
            set_three_term_row(
                row,
                glp.GLP_UP,
                0.0,
                2.0,
                (y, fi, fj),
                (1.0, 1.0, 1.0),
            )
            row += 1

        params = glp.glp_iocp()
        glp.glp_init_iocp(params)

        params.presolve = glp.GLP_ON
        params.msg_lev = glp.GLP_MSG_ALL if log else glp.GLP_MSG_OFF
        params.tm_lim = max(0, int(float(time_limit_s) * 1000))

        return_code = glp.glp_intopt(prob, params)
        mip_status = glp.glp_mip_status(prob)

        status_names = {
            glp.GLP_OPT: "OPTIMAL",
            glp.GLP_FEAS: "FEASIBLE",
            glp.GLP_NOFEAS: "INFEASIBLE",
            glp.GLP_UNDEF: "UNDEFINED",
        }
        status = status_names.get(mip_status, f"GLPK_STATUS_{mip_status}")

        if mip_status in (glp.GLP_OPT, glp.GLP_FEAS):
            flips = bitstring.tensor(
                [int(round(glp.glp_mip_col_val(prob, f_col(i)))) for i in range(n)],
            )
            objective_value = float(glp.glp_mip_obj_val(prob))

            if return_code == glp.GLP_ETMLIM and mip_status == glp.GLP_FEAS:
                status = "TIME_LIMIT_FEASIBLE"
        else:
            flips = bitstring.zeros(n)
            objective_value = float("nan")

            if return_code == glp.GLP_ETMLIM:
                status = "TIME_LIMIT_NO_SOLUTION"
            elif return_code != 0:
                status = f"GLPK_ERROR_{return_code}"

            hint = (
                " Consider increasing time_limit_s to give GLPK more time to find a solution."
                if status == "TIME_LIMIT_NO_SOLUTION"
                else ""
            )
            logger.warning(
                f"Bit-flip preprocessing ILP did not reach a feasible solution "
                f"(status={status}); falling back to no-op flips.{hint}"
            )

    except Exception:
        logger.warning(
            "Bit-flip preprocessing ILP raised an exception; falling back to no-op flips.",
            exc_info=True,
        )
        flips = bitstring.zeros(n)
        status = "FAIL"
        objective_value = float("nan")

    finally:
        glp.glp_delete_prob(prob)

    return flips, objective_value, status


class Instance(qubosolver.Instance):
    """A QUBO instance carrying bit-flip preprocessing history.

    Wraps a parent [`qubosolver.Instance`][] whose off-diagonal coefficients may
    contain negative interactions.  Applying [`apply`][] solves the bit-flip ILP,
    stores the flip vector here, and exposes the transformed matrix so it can be
    embedded and solved.  [`unapply`][] uses the stored state to map a solution
    back onto the original variables.
    """

    def __init__(self, parent_instance: qubosolver.Instance):
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original QUBO instance (before bit flips).
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(parent_instance.matrix.detach().clone())
        self._parent_instance = copy.deepcopy(parent_instance)
        self.flips: Bitstring = bitstring.zeros(parent_instance.size)
        self.metrics: dict[str, Any] = {}
        self.status: str = "NONE"
        self.offset: float = 0.0


def apply(
    qubo: qubosolver.Instance,
    *,
    time_limit_s: float = 60.0,
    eps: float = 0.0,
) -> Instance:
    """Solve the bit-flip ILP and apply the optimal flips to the QUBO matrix.

    Wraps *qubo* in a bit-flip [`Instance`][], solves the negative-weight ILP
    with GLPK, and replaces the matrix with its flipped counterpart.  When
    *qubo* has no negative off-diagonal coefficient, the wrapper is returned
    unchanged (``status`` stays ``"NONE"`` and ``flips`` stays all-zero).  If
    the solved flips would leave *more* negative weight than doing nothing
    (a known GLPK edge case), they are rejected and replaced with a no-op
    (``status`` becomes ``"REJECTED_WORSE_THAN_NOOP"``).

    Args:
        qubo: The QUBO instance to preprocess.
        time_limit_s: GLPK solver time limit in seconds.
        eps: Tolerance below which a coefficient is treated as zero.

    Returns:
        A bit-flip [`Instance`][] carrying the flip vector and metrics.
    """
    instance = Instance(qubo)

    Q = qubo.matrix
    if instance.size == 0 or not _has_negative_offdiagonal(Q, eps=eps):
        return instance

    flips, objective_value, status = _solve_bitflip_preprocessing_glpk(
        Q,
        time_limit_s=time_limit_s,
        eps=eps,
    )
    metrics = _compute_negative_weight_metrics(Q, flips, eps)

    if metrics["neg_weight_reduction_pct"] < 0.0:
        reduction_pct = metrics["neg_weight_reduction_pct"]
        logger.warning(
            f"Bit-flip preprocessing (status={status}) increased the remaining negative "
            f"off-diagonal weight instead of reducing it ({reduction_pct:.2f}% change); "
            f"falling back to no-op flips."
        )
        flips.fill_(0)
        objective_value = float("nan")
        status = "REJECTED_WORSE_THAN_NOOP"
        metrics = _compute_negative_weight_metrics(Q, flips, eps)

    Q_flipped, offset = _transform_qubo_with_bitflips(Q, flips)

    instance._matrix = Q_flipped
    instance.flips = flips
    instance.metrics = metrics
    instance.metrics["objective_value"] = objective_value
    instance.status = status
    instance.offset = offset

    return instance


def unapply(flipped_solution: Solution, flipped_qubo: Instance) -> Solution:
    """Map a solution of the bit-flipped QUBO back onto the original variables.

    Undoes the flips recorded on *flipped_qubo* (``y_i -> x_i``) and recomputes
    costs against the original (unflipped) matrix.  When no bit flip was applied,
    returns a deep copy of *flipped_solution* unchanged.

    Args:
        flipped_solution: Solution obtained on the bit-flipped QUBO.
        flipped_qubo: The bit-flip [`Instance`][] produced by [`apply`][].

    Returns:
        A new solution over the original variables.
    """
    flipped = torch.any(flipped_qubo.flips != 0)
    if not flipped:
        return copy.deepcopy(flipped_solution)

    solution = Solution()
    solution.bitstrings = _apply_bitflips(flipped_solution.bitstrings, flipped_qubo.flips)
    solution.costs = vector.tensor(
        [flipped_qubo._parent_instance.evaluate_solution(b) for b in solution.bitstrings]
    )
    solution.counts = flipped_solution.counts
    solution.probabilities = flipped_solution.probabilities

    return solution
