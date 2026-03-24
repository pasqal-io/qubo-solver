"""Reusable utilities: summary + binary visualizations (no heatmap, 2 colors)"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qubosolver.data import QUBOSolution
from qubosolver.qubo_analyzer import QUBOAnalyzer


def feasibility_violation(vals: list[float], A: np.ndarray, tol: float = 1e-9) -> int:
    """
    Count the numbers of items not covered.
    We consider 'covered' iff cover > tol.
    """
    x = np.array(vals, dtype=float)
    cover = A @ x
    return int(np.sum(cover <= tol))


def bitstring_from_vals(vals: list[float]) -> str:
    """vals in same order as `subsets`"""
    return "".join(str(int(round(v))) for v in vals)


def chosen_list(vals: list[float], subsets: list[str]) -> list[str]:
    return [subsets[p] for p, v in enumerate(vals) if int(round(v)) == 1]


def coverage_report(
    vals: list[float],
    subsets: list[str],
    items: list[str],
    nb_items: int,
    nb_partitions: int,
    A: np.ndarray,
) -> str:
    lines = []
    for i in range(nb_items):
        covering = [
            subsets[p] for p in range(nb_partitions) if A[i, p] == 1 and int(round(vals[p])) == 1
        ]
        lines.append(f"  Item {items[i]} covered by: {covering if covering else '—'}")
    return "\n".join(lines)


def bitstring_from_x(x: list[float | int] | np.ndarray) -> str:
    x = np.asarray(x).astype(int).tolist()
    return "".join(str(b) for b in x)


def selected_names(x: list[float | int] | np.ndarray, subsets: list[str]) -> list:
    x = np.asarray(x).astype(int)
    return [subsets[i] for i, b in enumerate(x) if b == 1]


def cover_vector(x: list[float | int] | np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return coverage counts c = A x (how many times each item is covered)."""
    x = np.asarray(x, dtype=float)
    return A @ x


def exact_violation(x: list[float | int] | np.ndarray, A: np.ndarray) -> int:
    """Exact-cover violation: sum_i |(Ax)_i - 1|."""
    c = cover_vector(x, A)
    return int(np.sum(np.abs(c - 1.0)))


def ilp_objective(x: list[float | int] | np.ndarray, w: np.ndarary) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.dot(w, x))


def qubo_objective(
    x: list[float | int] | np.ndarray,
    Q: list[float | int] | np.ndarray,
    const_offset: float = 0.0,
) -> float:
    x = np.asarray(x, dtype=float)
    Qnp = Q.cpu().numpy() if hasattr(Q, "cpu") else np.asarray(Q, dtype=float)
    return float(x @ Qnp @ x + const_offset)


# ---- Binary visuals (two colors only) ----
def show_binary_status(
    subsets: list[str],
    x: list[float | int] | np.ndarray,
    items: list[str],
    A: np.ndarray,
    *,
    item_rule: str = "at_least_one",
    figsize: tuple[int, float] = (7, 2.8),
) -> None:
    """
    Two binary rows:
      - row 0: subsets selected (1) vs not (0)
      - row 1: items covered (1) vs not (0), by rule:
          'at_least_one'  -> covered if (Ax)_i >= 1
          'exactly_one'   -> covered if (Ax)_i == 1
    """
    x = np.asarray(x).astype(int)
    P = len(subsets)
    c = cover_vector(x, A)

    if item_rule == "exactly_one":
        item_mask = (np.abs(c - 1.0) < 1e-9).astype(int)
        item_title = "items covered (exactly once)"
    else:
        item_mask = (c >= 1.0 - 1e-9).astype(int)
        item_title = "items covered (≥ 1)"

    subset_mask = x.copy()

    # Build a 2 x max(P, |I|) binary panel (pad with -1s to equal widths)
    W = max(P, A.shape[0])
    panel = -np.ones((2, W), dtype=int)
    panel[0, :P] = subset_mask
    panel[1, : A.shape[0]] = item_mask

    # two-color palette: 0 -> light gray, 1 -> dark (no colorbar)
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#d0d0d0", "#303030"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(panel, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["subsets selected", item_title])

    # x-ticks: names for subsets on row 0; item indices on row 1
    xticks = range(W)
    xticklabels = []
    for j in xticks:
        if j < P:
            xticklabels.append(subsets[j])
        else:
            xticklabels.append("")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)

    # Small grid separation
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.8)
    ax.set_title("Binary status: subsets (selected) & items (covered)")

    # Second axis on top to annotate item indices over the second row
    ax_top = ax.secondary_xaxis("top")
    top_labels = []
    for j in xticks:
        if j < A.shape[0]:
            top_labels.append(f"item {j+1}")
        else:
            top_labels.append("")
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(top_labels)
    plt.tight_layout()
    plt.show()


def summarize_solution(
    x: list[float | int] | np.ndarray,
    subsets: list[str],
    A: np.ndarray,
    w: np.ndarray,
    Q: list[float | int] | np.ndarray | None = None,
    const_offset: float = 0.0,
    best_ilp_cost: float | None = None,
    label: str = "Solution",
) -> None:
    """
    Print a concise summary + feasibility and (if feasible) gap to ILP optimum.
    Gap convention (your choice): (best - current) / best * 100.
    => 0% at optimum; negative if the feasible solution is worse than optimum.
    """
    x = np.asarray(x).astype(int)
    bitstr = bitstring_from_x(x)
    selected = selected_names(x, subsets)
    ilp_val = ilp_objective(x, w)
    viol = exact_violation(x, A)
    feasible = viol == 0

    print(f"=== {label} ===")
    print("Optimal bitstring (order: {}): {}".format(", ".join(subsets), bitstr))
    print("Selected variables            :", selected if selected else "∅")
    print(f"ILP objective (Σ w_p x_p)     : {ilp_val:g}")
    print(f"Feasibility (exact cover)     : \
        {'Feasible ✅' if feasible else f'Infeasible ❌ (violation={viol})'}")

    if Q is not None:
        qobj = qubo_objective(x, Q, const_offset=const_offset)
        print(f"QUBO objective (x^T Q x + cst): {qobj:g}")

    # Gap only if feasible and best_ilp_cost provided
    if best_ilp_cost is not None:
        if feasible:
            gap = (ilp_val - best_ilp_cost) / best_ilp_cost * 100.0
            print(f"Gap to ILP optimum           : {gap:.2f}%  (opt = {best_ilp_cost:g})")
        else:
            print("Gap to ILP optimum           : N/A (infeasible)")


# --- Pretty polynomial (constant / linear / quadratic) ---
def _fmt(c: float) -> str:
    if abs(c - round(c)) < 1e-12:
        return str(int(round(c)))
    return f"{c:.6g}"


def _join_terms(terms: list = list()) -> str:
    terms = [(c, lab) for (c, lab) in terms if abs(c) > 1e-12]
    if not terms:
        return "0"
    out = []
    for k, (c, lab) in enumerate(terms):
        mag = _fmt(abs(c))
        if k == 0:
            out.append(f"{'-' if c < 0 else ''}{mag} {lab}")
        else:
            out.append((" + " if c > 0 else " - ") + f"{mag} {lab}")
    return "".join(out)


def compare_solutions(solutions: list[QUBOSolution], labels: list[str]) -> None:
    """Use QUBOAnalyzer to compare solutions via plotting."""
    analyzer = QUBOAnalyzer(solutions, labels=labels)
    analyzer.plot(
        x_axis="costs",
        y_axis="probs",
        sort_by="costs",
        sort_order="ascending",
        context="notebook",
    )
