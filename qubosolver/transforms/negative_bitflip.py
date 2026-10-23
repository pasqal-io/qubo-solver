"""Bit-flip preprocessing transforms for QUBO instances with negative interactions.

Quantum (Rydberg) solvers cannot encode attractive interactions, so a QUBO must
have non-negative off-diagonal coefficients to be embeddable.  A change of
variable ``x_i -> 1 - y_i`` on a subset of variables flips the sign of the
interactions incident to it; choosing the subset that removes as much negative
weight as possible is itself a QUBO, ``f^T Q_f f`` with ``Q_f = diag(C 1) - C``
(see [`bitflip_qubo`][]), solved here by local search, tabu search or, as an
integer linear program, with [GLPK](https://github.com/bradfordboyle/pyglpk).

[`apply`][] finds the bit flips (by default with [`solve_bitflip_local_search`][]) and
applies them to the matrix, returning a wrapper [`Instance`][] that
records the flip vector so the solution can later be mapped back with
[`lift`][].  When bit flips
cannot remove *every* negative off-diagonal coefficient, the remaining ones can
be dropped with [`transforms.zeroing.apply`][].

Typical usage:

```python
from qubosolver.transforms import negative_bitflip
from qubosolver.solving import brute_force

flipped_instance = negative_bitflip.apply(instance, time_limit_s=60.0)
flipped_solution = brute_force.solve(flipped_instance)
solution = negative_bitflip.lift(flipped_solution, flipped_instance)
```
"""

from __future__ import annotations

import copy
import io
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch

import qubosolver
from qubosolver._checks import TYPE_CHECKING
from qubosolver._io import utils as io_utils
from qubosolver.solving.classical import tabu_search
from qubosolver.solving.classical.iterative_bitflip_local_search import (
    _best_improvement_search_batch,
)
from qubosolver.types import (
    Bitstring,
    Bitstrings,
    Matrix,
    Solution,
    bitstring,
    bitstrings,
    matrix,
    vector,
)

if TYPE_CHECKING:
    from qubosolver._io.utils import Stream
    from qubosolver.types.linalg import Matrixd

logger = logging.getLogger(__name__)


def _import_swiglpk() -> Any:  # noqa: ANN401 (SWIG module has no type stubs)
    """Import ``swiglpk`` with its import-time SWIG warnings silenced.

    SWIG-generated extensions emit ``DeprecationWarning: builtin type ... has
    no __module__ attribute`` while initializing on Python 3.12+.  If warnings
    are turned into errors (``-W error``, or pytest's ``filterwarnings =
    ["error"]``), that warning aborts the extension's initialization half-way
    and the next GLPK call segfaults.  Once imported, the module is cached in
    ``sys.modules`` so later imports are safe.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import swiglpk  # noqa: PLC0415 (imported with SWIG deprecation warnings silenced)

    return swiglpk


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
        "neg_count_reduction_pct": reduction_pct(float(neg_count_before), float(neg_count_after)),
        "neg_weight_before": neg_weight_before,
        "neg_weight_after": neg_weight_after,
        "neg_weight_reduction_pct": reduction_pct(neg_weight_before, neg_weight_after),
    }


def _solve_bitflip_preprocessing_glpk(  # noqa: C901, PLR0915
    # This function is planned to be reimplemented without GLPK; complexity
    # should be addressed as part of that rewrite rather than fixed here.
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
    glp = _import_swiglpk()

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

        def set_three_term_row(  # noqa: PLR0913, PLR0917
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
                zip(columns, coefficients, strict=True),
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
            # GLPK's default integrality tolerance is 1e-5, looser than
            # `round`'s default.
            flips = bitstring.round(
                [glp.glp_mip_col_val(prob, f_col(i)) for i in range(n)],
                atol=1e-4,
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
                "Bit-flip preprocessing ILP did not reach a feasible solution "
                "(status=%s); falling back to no-op flips.%s",
                status,
                hint,
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


def bitflip_qubo(Q: Matrix, *, eps: float = 0.0) -> Matrixd | None:
    """Build the QUBO ``Q_f`` whose minimum gives the best bit flips for `Q`.

    The bit-flip problem picks flips ``f`` minimizing the negative off-diagonal
    weight left after ``x_i -> 1 - x_i``, i.e. ``sum_{i<j} c_ij (f_i XOR f_j)``.
    Since ``f_i XOR f_j = f_i + f_j - 2 f_i f_j``, this is itself the QUBO
    ``f^T Q_f f`` with ``Q_f = diag(C 1) - C``.

    Args:
        Q: QUBO coefficient matrix; only the upper triangle is read.
        eps: Off-diagonal coefficients with ``abs(q) <= eps`` are ignored.

    Returns:
        ``Q_f`` in float64, or ``None`` when no off-diagonal coefficient
        survives `eps`.
    """
    n = Q.shape[0]
    C = torch.triu(Q.detach().to(torch.float64), diagonal=1)
    C = torch.where(C.abs() > eps, C, torch.zeros_like(C))
    C = C + C.T

    if n == 0 or not C.any():
        return None

    return torch.diag(C.sum(dim=1)) - C


def solve_bitflip_local_search(
    Q: Matrix,
    *,
    eps: float = 0.0,
    n_starts: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, float, str]:
    """Find bit flips for `Q` by multistart local search on [`bitflip_qubo`][].

    The starts are the no-op plus ``n_starts - 1`` random ones, so the result
    is never worse than no flips.

    Args:
        Q: QUBO coefficient matrix; only the upper triangle is read.
        eps: Off-diagonal coefficients with ``abs(q) <= eps`` are ignored.
        n_starts: Number of starts, the no-op plus ``n_starts - 1`` random ones.
        seed: Seed of the random starts, for reproducibility.

    Returns:
        The flip vector, the objective value and ``"FEASIBLE"`` (a local
        optimum, not a proven one), or ``"OPTIMAL"`` with no flips when there
        is nothing to flip.
    """
    n = Q.shape[0]
    Q_f = bitflip_qubo(Q, eps=eps)
    if Q_f is None:
        return bitstring.zeros(n), 0.0, "OPTIMAL"

    generator = torch.Generator(device=Q_f.device).manual_seed(seed)
    F = bitstrings.zeros(max(1, n_starts), n, device=Q_f.device)
    F[1:] = torch.randint(
        0, 2, (F.shape[0] - 1, n), generator=generator, device=Q_f.device, dtype=F.dtype
    )
    # The search runs in the library dtype; objectives are recomputed in float64.
    F = _best_improvement_search_batch(matrix.as_tensor(Q_f), F).to(Q_f.dtype)

    objectives = ((F @ Q_f) * F).sum(dim=1)
    best = int(objectives.argmin())
    flips = F[best].to(bitstring.dtype()).to(Q.device)
    return flips, float(objectives[best]), "FEASIBLE"


def solve_bitflip_tabu_search(  # noqa: PLR0913
    Q: Matrix,
    *,
    eps: float = 0.0,
    n_starts: int = 100,
    seed: int = 0,
    max_iter: int = 1000,
    tabu_tenure: int = 7,
    max_no_improve: int = 100,
    time_limit_s: float = float("inf"),
) -> tuple[torch.Tensor, float, str]:
    """Find bit flips for `Q` by local search followed by tabu search.

    Runs the same multistart local search as [`solve_bitflip_local_search`][],
    then every local optimum starts one tabu run on [`bitflip_qubo`][], which
    walks out of it through non-improving moves. Each run keeps its best
    point, so the result is never worse than the local search's.

    Args:
        Q: QUBO coefficient matrix; only the upper triangle is read.
        eps: Off-diagonal coefficients with ``abs(q) <= eps`` are ignored.
        n_starts: Number of starts, the no-op plus ``n_starts - 1`` random ones.
        seed: Seed of the random starts, for reproducibility.
        max_iter: Maximum number of tabu iterations.
        tabu_tenure: Number of iterations a flipped bit stays tabu.
        max_no_improve: The search stops once every run has gone this many
            iterations without improving its best.
        time_limit_s: Wall-clock budget of the tabu search, in seconds.

    Returns:
        The flip vector, the objective value and ``"FEASIBLE"`` (not a proven
        optimum), or ``"OPTIMAL"`` with no flips when there is nothing to flip.
    """
    n = Q.shape[0]
    Q_f = bitflip_qubo(Q, eps=eps)
    if Q_f is None:
        return bitstring.zeros(n), 0.0, "OPTIMAL"

    generator = torch.Generator(device=Q_f.device).manual_seed(seed)
    F = bitstrings.zeros(max(1, n_starts), n, device=Q_f.device)
    F[1:] = torch.randint(
        0, 2, (F.shape[0] - 1, n), generator=generator, device=Q_f.device, dtype=F.dtype
    )
    # The search runs in the library dtype; objectives are recomputed in float64.
    F = _best_improvement_search_batch(matrix.as_tensor(Q_f), F)

    solution = tabu_search.solve(
        qubosolver.Instance(matrix.as_tensor(Q_f)),
        starts=F,
        max_iter=max_iter,
        tabu_tenure=tabu_tenure,
        max_no_improve=max_no_improve,
        time_limit=time_limit_s,
    )

    # The solver's costs are in the library dtype; recompute in float64.
    F = solution.bitstrings.to(Q_f.dtype)
    objectives = ((F @ Q_f) * F).sum(dim=1)
    best = int(objectives.argmin())
    flips = solution.bitstrings[best].to(bitstring.dtype()).to(Q.device)
    return flips, float(objectives[best]), "FEASIBLE"


class Instance(qubosolver.Instance):
    """A QUBO instance carrying bit-flip preprocessing history.

    Wraps a parent [`qubosolver.Instance`][] whose off-diagonal coefficients may
    contain negative interactions.  Applying [`apply`][] solves the bit-flip ILP,
    stores the flip vector here, and exposes the transformed matrix so it can be
    embedded and solved.  [`lift`][] uses the stored state to map a solution
    back onto the original variables.
    """

    def __init__(self, parent_instance: qubosolver.Instance) -> None:
        """Initialize from a parent QUBO instance.

        Args:
            parent_instance: The original QUBO instance (before bit flips).
                A deep copy is kept internally for later reconstruction.
        """
        super().__init__(parent_instance.matrix.detach().clone())
        self._parent_instance = copy.deepcopy(parent_instance)

        self.flips: Bitstring = bitstring.zeros(parent_instance.size)
        """Flip vector applied to the parent matrix, all-zero until [`apply`][] populates it."""

        self.metrics: dict[str, Any] = {}
        """Negative off-diagonal count and weight before/after the flips, as computed
        by [`apply`][]."""

        self.status: str = "NONE"
        """Outcome of the bit-flip ILP solve (e.g. ``"NONE"``, ``"OPTIMAL"``,
        ``"REJECTED_WORSE_THAN_NOOP"``)."""

        self.offset: float = 0.0
        """Constant term relating the flipped and original QUBO costs,
        $x^T Q x = y^T Q_{flipped} y + offset$."""

    def _write_body(self, f: Stream[bytes]) -> None:
        """Write the matrix, flip vector, solve metadata, and parent instance to `f`."""
        super()._write_body(f)

        buffer = io.BytesIO()
        torch.save(self.flips, buffer)
        io_utils.save_sized_buffer(f, buffer.getbuffer())

        state_json = json.dumps(
            {"status": self.status, "offset": self.offset, "metrics": self.metrics}
        )
        io_utils.save_string(f, state_json)

        self._parent_instance.save(f)

    @classmethod
    def _read_body(cls, f: Stream[bytes]) -> Instance:
        """Read back a negative-bitflip instance written by [`_write_body`][]."""
        instance = Instance(qubosolver.Instance._read_body(f))

        buffer = io.BytesIO(io_utils.load_sized_buffer(f))
        instance.flips = torch.load(buffer, weights_only=True)

        state = json.loads(io_utils.load_string(f))
        instance.status = state["status"]
        instance.offset = state["offset"]
        instance.metrics = state["metrics"]

        instance._parent_instance = qubosolver.Instance.load(f)
        return instance


def apply(
    instance: qubosolver.Instance,
    *,
    method: Literal["local_search", "tabu_search", "glpk"] = "local_search",
    time_limit_s: float = 60.0,
    eps: float = 0.0,
) -> Instance:
    """Find the bit flips minimizing negative weight and apply them to the QUBO matrix.

    Wraps `instance` in a bit-flip [`Instance`][], finds the flips with `method`,
    and replaces the matrix with its flipped counterpart.  When
    `instance` has no negative off-diagonal coefficient, the wrapper is returned
    unchanged (``status`` stays ``"NONE"`` and ``flips`` stays all-zero).  If
    the solved flips would leave *more* negative weight than doing nothing
    (a known `GLPK` edge case), they are rejected and replaced with a no-op
    (``status`` becomes ``"REJECTED_WORSE_THAN_NOOP"``).

    Args:
        instance: The QUBO instance to preprocess.
        method: ``"local_search"`` ([`solve_bitflip_local_search`][]),
            ``"tabu_search"`` ([`solve_bitflip_tabu_search`][]) or ``"glpk"``
            (the **Integer Linear Program (ILP)** solved with
            [`GLPK`](https://github.com/bradfordboyle/pyglpk)).
        time_limit_s: Time limit in seconds of the tabu search or `GLPK`;
            unused by the local search.
        eps: Tolerance below which a coefficient is treated as zero.

    Returns:
        A bit-flip instance carrying the flip vector and metrics.

    Raises:
        ValueError: If `method` is unknown.
    """
    flipped_instance = Instance(instance)

    Q = instance.matrix
    if instance.size == 0 or not _has_negative_offdiagonal(Q, eps=eps):
        return flipped_instance

    if method == "local_search":
        flips, objective_value, status = solve_bitflip_local_search(Q, eps=eps)
    elif method == "tabu_search":
        flips, objective_value, status = solve_bitflip_tabu_search(
            Q, eps=eps, time_limit_s=time_limit_s
        )
    elif method == "glpk":
        flips, objective_value, status = _solve_bitflip_preprocessing_glpk(
            Q,
            time_limit_s=time_limit_s,
            eps=eps,
        )
    else:
        msg = f"Unknown bit-flip method {method!r}."
        raise ValueError(msg)
    metrics = _compute_negative_weight_metrics(Q, flips, eps)

    if metrics["neg_weight_reduction_pct"] < 0.0:
        reduction_pct = metrics["neg_weight_reduction_pct"]
        logger.warning(
            "Bit-flip preprocessing (status=%s) increased the remaining negative "
            "off-diagonal weight instead of reducing it (%.2f%% change); "
            "falling back to no-op flips.",
            status,
            reduction_pct,
        )
        flips.fill_(0)
        objective_value = float("nan")
        status = "REJECTED_WORSE_THAN_NOOP"
        metrics = _compute_negative_weight_metrics(Q, flips, eps)

    Q_flipped, offset = _transform_qubo_with_bitflips(Q, flips)

    flipped_instance._matrix = Q_flipped
    flipped_instance.flips = flips
    flipped_instance.metrics = metrics
    flipped_instance.metrics["objective_value"] = objective_value
    flipped_instance.status = status
    flipped_instance.offset = offset

    return flipped_instance


def lift(flipped_solution: Solution, flipped_instance: Instance) -> Solution:
    """Map a solution of the bit-flipped QUBO instance back onto the original variables.

    Undoes the flips recorded on `flipped_instance` (``y_i -> x_i``) and recomputes
    costs against the original (unflipped) matrix.  When no bit flip was applied,
    returns a deep copy of `flipped_solution` unchanged.

    Args:
        flipped_solution: Solution obtained on the bit-flipped instance.
        flipped_instance: The bit-flipped instance produced by [`apply`][].

    Returns:
        A new solution over the original variables.
    """
    flipped = torch.any(flipped_instance.flips != 0)
    if not flipped:
        return copy.deepcopy(flipped_solution)

    solution = Solution()
    solution.bitstrings = _apply_bitflips(flipped_solution.bitstrings, flipped_instance.flips)
    solution.costs = vector.tensor(
        [flipped_instance._parent_instance.cost(b) for b in solution.bitstrings]
    )
    solution.counts = flipped_solution.counts
    solution.probabilities = flipped_solution.probabilities

    return solution
