from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Edge:
    """Non-zero upper-triangular QUBO interaction."""

    i: int
    j: int
    q: float


def has_negative_offdiagonal(Q: torch.Tensor, eps: float = 0.0) -> bool:
    """Return True if Q has at least one negative off-diagonal coefficient."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=Q.device)
    return bool(torch.any(Q[offdiag_mask] < -eps).item())


def extract_edges_upper(Q: torch.Tensor, eps: float = 0.0) -> list[Edge]:
    """Extract non-zero upper-triangular off-diagonal QUBO coefficients."""
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    Q_cpu = Q.detach().cpu()
    n = Q_cpu.shape[0]
    edges: list[Edge] = []

    for i in range(n):
        for j in range(i + 1, n):
            q = float(Q_cpu[i, j].item())
            if abs(q) > eps:
                edges.append(Edge(i=i, j=j, q=q))

    return edges


def apply_bitflips_to_bitstrings(
    bitstrings: torch.Tensor,
    flips: torch.Tensor,
) -> torch.Tensor:
    """Apply or undo bit flips on bitstrings.

    The operation is its own inverse.
    """
    if bitstrings.numel() == 0:
        return bitstrings

    flips = flips.to(device=bitstrings.device, dtype=bitstrings.dtype)

    if bitstrings.ndim == 1:
        return torch.abs(bitstrings - flips)

    if bitstrings.ndim == 2:
        return torch.abs(bitstrings - flips.unsqueeze(0))

    raise ValueError("bitstrings must be a 1D or 2D tensor.")


def transform_qubo_by_bitflips(
    Q: torch.Tensor,
    flips: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Transform QUBO coefficients after variable bit flips.

    Convention:
        x_i = y_i       if flips_i = 0
        x_i = 1 - y_i   if flips_i = 1

    The returned offset satisfies:
        x^T Q x = y^T Q_flipped y + offset
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q.shape[0]
    if flips.numel() != n:
        raise ValueError("flips must have the same length as Q size.")

    dtype = Q.dtype
    device = Q.device

    f = flips.to(device=device, dtype=dtype).reshape(n)
    s = 1.0 - 2.0 * f

    Q_flipped = Q * torch.outer(s, s)

    linear = 2.0 * s * (Q @ f)
    Q_flipped = Q_flipped.clone()
    diag_idx = torch.arange(n, device=device)
    Q_flipped[diag_idx, diag_idx] += linear

    offset = float(f @ Q @ f)

    return Q_flipped, offset


def _coefficient_after_bitflip(q: float, fi: int, fj: int) -> float:
    """Return the off-diagonal coefficient sign after bit flips."""
    xor = fi ^ fj
    return (1.0 - 2.0 * xor) * q


def compute_negative_weight_metrics(
    Q: torch.Tensor,
    flips: torch.Tensor,
    eps: float = 0.0,
) -> dict[str, Any]:
    """Compute negative off-diagonal count and weight before and after bit flips."""
    flips_list = [int(v) for v in flips.detach().cpu().tolist()]
    edges = extract_edges_upper(Q, eps=eps)

    neg_count_before = 0
    neg_count_after = 0
    neg_weight_before = 0.0
    neg_weight_after = 0.0

    for edge in edges:
        q = edge.q

        if q < -eps:
            neg_count_before += 1
            neg_weight_before += abs(q)

        q_after = _coefficient_after_bitflip(q, flips_list[edge.i], flips_list[edge.j])
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


def solve_bitflip_preprocessing_glpk(
    Q: torch.Tensor,
    *,
    time_limit_s: float = 10.0,
    eps: float = 0.0,
    log: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    """Solve the GLPK bit-flip preprocessing ILP with the negative-weight objective.

    Variables:
        f_i in {0, 1}
        y_k in {0, 1}, with y_k = f_i XOR f_j

    Objective:
        minimize the total absolute weight of negative off-diagonal coefficients
        remaining after applying the variable flips.
    """
    import swiglpk as glp

    Q_cpu = Q.detach().cpu()
    n = int(Q_cpu.shape[0])
    edges = extract_edges_upper(Q_cpu, eps=eps)
    m = len(edges)

    def f_col(i: int) -> int:
        return i + 1

    def y_col(k: int) -> int:
        return n + k + 1

    if m == 0:
        flips = torch.zeros(n, dtype=torch.int64)
        metrics = compute_negative_weight_metrics(Q_cpu, flips, eps=eps)
        metrics["objective_value"] = 0.0
        return flips, metrics, "OPTIMAL"

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
            flips = torch.tensor(
                [int(round(glp.glp_mip_col_val(prob, f_col(i)))) for i in range(n)],
                dtype=torch.int64,
            )
            objective_value = float(glp.glp_mip_obj_val(prob))

            if return_code == glp.GLP_ETMLIM and mip_status == glp.GLP_FEAS:
                status = "TIME_LIMIT_FEASIBLE"
        else:
            flips = torch.zeros(n, dtype=torch.int64)
            objective_value = float("nan")

            if return_code == glp.GLP_ETMLIM:
                status = "TIME_LIMIT_NO_SOLUTION"
            elif return_code != 0:
                status = f"GLPK_ERROR_{return_code}"

    except Exception:
        flips = torch.zeros(n, dtype=torch.int64)
        status = "FAIL"
        objective_value = float("nan")

    finally:
        glp.glp_delete_prob(prob)

    metrics = compute_negative_weight_metrics(Q_cpu, flips, eps=eps)
    metrics["objective_value"] = objective_value

    return flips, metrics, status