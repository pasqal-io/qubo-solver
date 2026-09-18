"""Simulated Annealing solver for QUBO problems.

Implements a bit-flip annealer that minimizes the quadratic objective
$E(x) = x^T Q x$ over binary vectors $x \\in \\{0,1\\}^n$, run independently from
each of a batch of starting points and merged into a single solution.
"""

from __future__ import annotations

import time
import heapq
import logging
from dataclasses import dataclass
from typing import Literal, overload

import torch

from qubosolver import (
    Instance,
    Solution,
    bitstring,
    bitstrings,
    vector,
    Bitstring,
    Bitstrings,
    torch_rng,
    vectori,
)

from qubosolver.utils._costs import batched_quadratic_cost, _flip_deltas

logger = logging.getLogger(__name__)


@dataclass
class _Data:
    energy: float = float("inf")
    count: int = 0


def _to_key(bits: Bitstring) -> bytes:
    return bytes(bits.tolist())


def _from_key(key: bytes) -> Bitstring:
    # Bitstring is always torch.int8, so this round-trips exactly through
    # the same bytes produced by _to_key.
    return torch.frombuffer(bytearray(key), dtype=bitstring.dtype())


def _item_energy(item: tuple[bytes, _Data]) -> float:
    return item[1].energy


def _shrink(visited_solutions: dict[bytes, _Data], *, top_k: int) -> None:
    # Mutates visited_solutions in place (no return value) so callers keep
    # their reference valid instead of having to reassign it.
    kept = heapq.nsmallest(top_k, visited_solutions.items(), key=_item_energy)
    visited_solutions.clear()
    visited_solutions.update(kept)


# Built once at import time and shared as the default `rng` across the
# overload stubs and the implementation below, so calls that omit `rng`
# consistently reuse the same generator regardless of which overload mypy
# picked, instead of each `def` capturing its own independent instance.
_default_rng = torch_rng()

# How often the vectorized runner recomputes `energy` and `QX` exactly instead
# of accumulating them incrementally. Small enough that rounding error stays
# well below the tolerance of `Solution.check_consistency`, large enough that
# the extra matmul stays negligible against the per-iteration cost.
_REFRESH_EVERY = 128


@overload
def solve(
    instance: Instance,
    *,
    starts: Bitstrings | int = 1,
    merge: Literal[True] = True,
    top_k: int = 1,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    time_limit: float = float("inf"),
    rng: torch.Generator = _default_rng,
    stats: Literal["per_run", "full"] = "per_run",
    vectorized: bool = True,
) -> Solution: ...


@overload
def solve(
    instance: Instance,
    starts: Bitstrings | int = 1,
    *,
    merge: Literal[False],
    top_k: int = 1,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    time_limit: float = float("inf"),
    rng: torch.Generator = _default_rng,
    stats: Literal["per_run", "full"] = "per_run",
    vectorized: bool = True,
) -> list[Solution]: ...


@torch.no_grad()
def solve(
    instance: Instance,
    starts: Bitstrings | int = 1,
    *,
    merge: bool = True,
    top_k: int = 1,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    time_limit: float = float("inf"),
    rng: torch.Generator = _default_rng,
    stats: Literal["per_run", "full"] = "per_run",
    vectorized: bool = True,
) -> Solution | list[Solution]:
    """Run Simulated Annealing on a QUBO instance from each of a batch of starting points.

    For each starting bitstring, at each of `max_iter` steps a random bit is
    proposed for flipping.  The flip is always accepted when it reduces the
    energy; otherwise it is accepted with probability $\\exp(-\\Delta E / T)$.

    Up to `top_k` unique lowest-energy bitstrings encountered during each run
    are retained, along with how many iterations were spent at each one
    (whether or not the proposed flip at that iteration was accepted). The
    runs are independent.  By default (``merge=True``) the per-start results
    are merged into a single [`Solution`][]. Pass ``merge=False`` to instead
    get back the unmerged, one-per-start list.

    Example:
        Running a single explicit start requires promoting it to a batch of
        size 1 first, via [`torch.unsqueeze`][]:

        ```python
        solution = simulated_annealing(instance, start.unsqueeze(0))
        ```

        Passing ``merge=False`` returns the per-start results instead of a
        single merged `Solution`:

        ```python
        solutions = simulated_annealing(instance, starts, merge=False)
        ```

    Args:
        instance: The QUBO instance to solve.  Its coefficient matrix is
            symmetrised internally as ``(Q + Qᵀ) / 2``.
        starts: Either a batch of initial binary solutions, a tensor of
            shape ``(k, n)`` with values in ``{0, 1}`` (one independent run
            is performed per row), or an ``int`` giving the number of
            uniformly random starts to generate.
        merge: When ``True`` (default), merge the per-start results into a
            single [`Solution`][]. When ``False``, return the unmerged list of one [`Solution`][] per
            starting point (same order as `starts`).
        top_k: Maximum number of unique best solutions to keep per run,
            ordered by ascending energy.
        max_iter: Number of bit-flip proposals to perform.
        initial_temp: Starting temperature $T_0$.  Higher values increase the
            probability of accepting uphill moves early in the search.
        final_temp: Target temperature $T_f$ at the end of the schedule, used
            to derive the cooling rate when `cooling_rate` is ``None``.
            Ignored when `cooling_rate` is provided explicitly.
        cooling_rate: Geometric cooling factor $\\alpha \\in (0, 1)$ such that
            $T \\leftarrow \\alpha T$ at each step.  When ``None`` (default),
            $\\alpha$ is derived automatically from `initial_temp`,
            `final_temp`, and `max_iter` so that the temperature reaches
            `final_temp` after `max_iter` steps.
        time_limit: Wall-clock budget in seconds.  The algorithm stops early
            when either `max_iter` steps or the time limit is reached,
            whichever comes first.  Defaults to ``float("inf")`` (no limit).
            With ``vectorized=True`` this is a single budget for the whole
            batch of starts, which all stop at the same iteration; with
            ``vectorized=False`` each start gets its own budget.
        rng: PyTorch random number generator used for bit selection and
            acceptance sampling.  Defaults to a module-level
            generator created once at import time; pass an explicit generator
            for reproducibility across calls.  Note that `vectorized` changes
            the order in which draws are consumed, so a given seed produces
            the same result only for a fixed value of `vectorized`.
        stats: When ``"per_run"`` (default), each run's retained bitstrings
            are counted as ``1`` instead of how many iterations were spent
            at each one, before any merging. This is mainly meant for
            ``top_k=1`` together with ``merge=True``, where the merged count
            directly reflects how many of the runs converged on each
            bitstring. With ``merge=True`` and ``top_k > 1``, or whenever a
            bitstring is retained by more than one run, [`deduplicate`][Solution.deduplicate] sums
            those per-run ``1``s, so counts on the merged result are
            generally neither ``1`` nor uniform. When ``"full"``, counts
            instead reflect how many iterations were spent at each
            bitstring.
        vectorized: When ``True`` (default), step every start forward together
            so each iteration costs a handful of batched tensor operations
            regardless of how many starts there are -- markedly faster for
            large batches. When ``False``, anneal the starts one at a time;
            this is the reference implementation, kept for comparison. The two
            run the same algorithm but differ in `time_limit` scope and in RNG
            draw order (see `time_limit` and `rng`).

    Returns:
        When ``merge=True``, a single [`Solution`][] merging every start's
            results.  When ``merge=False``, one [`Solution`][] per starting
            point (same order as `starts`).  Either way, each [`Solution`][]
            contains up to `top_k` unique bitstrings sorted by ascending energy,
            with their costs, counts (see `stats`), and probabilities.

    Raises:
        ValueError: If ``top_k < 1``.
        ValueError: If ``initial_temp <= 0``.
        ValueError: If ``cooling_rate`` is ``None`` and ``final_temp <= 0``.
        ValueError: If ``cooling_rate`` is provided but not in ``(0, 1)``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be >= 1.")
    if initial_temp <= 0:
        raise ValueError("initial_temp must be > 0.")
    if cooling_rate is None and final_temp <= 0:
        raise ValueError("final_temp must be > 0 when cooling_rate is None.")
    if stats == "per_run" and top_k > 1:
        logger.info(
            f"stats='per_run' with top_k={top_k}: per-run counts are set to 1, but merging "
            "sums them across runs, so merged counts are not simply 1 per run."
        )

    n = instance.size

    # determine cooling rate alpha
    if max_iter <= 1:
        alpha = 1.0
    elif cooling_rate is not None:
        alpha = float(cooling_rate)
        if not (0.0 < alpha < 1.0):
            raise ValueError("cooling_rate (alpha) must be in (0, 1).")
    else:
        alpha = (final_temp / initial_temp) ** (1.0 / (max_iter - 1))

    if isinstance(starts, int):
        starts = bitstrings.rand(starts, n, rng=rng)

    if starts.shape[0] == 0:
        return Solution() if merge else []

    runner = _run_vectorized if vectorized else _run_sequential
    solutions = runner(
        instance,
        starts,
        top_k=top_k,
        max_iter=max_iter,
        initial_temp=initial_temp,
        alpha=alpha,
        time_limit=time_limit,
        rng=rng,
        stats=stats,
    )

    if merge:
        return Solution.concat(solutions).deduplicate()

    return solutions


def _run_sequential(
    instance: Instance,
    starts: Bitstrings,
    *,
    top_k: int,
    max_iter: int,
    initial_temp: float,
    alpha: float,
    time_limit: float,
    rng: torch.Generator,
    stats: Literal["per_run", "full"],
) -> list[Solution]:
    """Anneal each start in turn, one scalar bit-flip proposal at a time.

    The original, run-at-a-time implementation, kept as the reference
    behaviour that [`_run_vectorized`][] is checked against. It is
    `O(len(starts) * max_iter)` in Python-level torch calls, so prefer the
    vectorized path for large batches.

    Unlike the vectorized path, `time_limit` here is consumed per run: each
    start gets its own fresh deadline.

    Args:
        instance: The QUBO instance to solve; its coefficient matrix must
            already be symmetric.
        starts: Batch of initial bitstrings of shape ``(k, n)``.
        top_k: Maximum number of unique best solutions to keep per run.
        max_iter: Number of bit-flip proposals per run.
        initial_temp: Starting temperature.
        alpha: Geometric cooling factor applied at each step.
        time_limit: Wall-clock budget in seconds, per run.
        rng: Generator used for bit selection and acceptance sampling.
        stats: See [`solve`][].

    Returns:
        One [`Solution`][] per row of `starts`, in the same order, each
            sorted by ascending cost with probabilities computed.
    """
    Q = instance.matrix
    n = Q.shape[0]
    solutions: list[Solution] = []

    for b in starts:
        bits: Bitstring = b.detach().clone()

        Qx = Q @ bits.to(Q)
        energy = float(bits.to(Q).dot(Qx))

        temperature: float = initial_temp

        visited_solutions: dict[bytes, _Data] = {}
        visited_solutions[_to_key(bits)] = _Data(energy, 1)

        deadline = time.perf_counter() + time_limit
        visits = 1

        for _ in range(max_iter):
            if time.perf_counter() >= deadline:
                break

            # See the matching comment in `_run_vectorized`: `energy` and `Qx`
            # are accumulated incrementally, so periodically recomputing them
            # exactly from `bits` keeps rounding error from growing unbounded.
            if visits % _REFRESH_EVERY == 0:
                Qx = Q @ bits.to(Q)
                energy = float(bits.to(Q).dot(Qx))

            i = int(torch.randint(0, n, (1,), generator=rng).item())
            xi = int(bits[i].item())

            # ΔE = (1 - 2xi) * (Q_ii + 2*(Qx_i - Q_ii*xi))
            Qii = float(Q[i, i].item())
            Qx_i = float(Qx[i].item())
            dE = (1 - 2 * xi) * (Qii + 2.0 * (Qx_i - Qii * xi))

            accept = (dE <= 0.0) or (
                torch.rand((), generator=rng).item()
                < torch.exp(torch.tensor(-dE / temperature)).item()
            )
            if accept:
                new_xi = 1 - xi
                diff = float(new_xi - xi)
                bits[i] = new_xi
                energy += dE
                Qx += diff * Q[:, i]

            key = _to_key(bits)
            sol = visited_solutions.setdefault(key, _Data(energy, 0))
            sol.count += 1
            visits += 1

            # Most inserts are one-off bitstrings that will never make the
            # top_k cut, so the dict keeps growing between shrinks regardless
            # of top_k. The "+100" gives it room to grow before paying for a
            # shrink; "2 *" just keeps that room proportional once top_k is
            # itself large (e.g. top_k=1000).
            limit = max(2 * top_k, top_k + 100)
            if len(visited_solutions) >= limit:
                _shrink(visited_solutions, top_k=top_k)

            temperature *= alpha
            if temperature < 1e-12:
                temperature = 1e-12

        _shrink(visited_solutions, top_k=top_k)

        counts = vectori.tensor([s.count for s in visited_solutions.values()])
        if stats == "per_run":
            counts.fill_(1)

        unique_bits = torch.stack([_from_key(key) for key in visited_solutions.keys()])
        solution = Solution(
            bitstrings=bitstrings.as_tensor(unique_bits),
            costs=vector.tensor([s.energy for s in visited_solutions.values()]),
            counts=counts,
        )

        # `costs` was accumulated incrementally, drifting from the true x^T Q x
        # by up to _REFRESH_EVERY steps of rounding error; recompute it exactly
        # now that the hot loop is done.
        solutions.append(solution._update(instance))

    return solutions


def _run_vectorized(
    instance: Instance,
    starts: Bitstrings,
    *,
    top_k: int,
    max_iter: int,
    initial_temp: float,
    alpha: float,
    time_limit: float,
    rng: torch.Generator,
    stats: Literal["per_run", "full"],
) -> list[Solution]:
    """Anneal every start simultaneously, one batched bit-flip proposal per step.

    All runs are stepped forward together: each iteration proposes one flip per
    run, evaluates every candidate delta in a single matmul, and accepts or
    rejects per run via a boolean mask. The number of Python-level torch calls
    is therefore proportional to `max_iter` alone rather than to
    ``len(starts) * max_iter``, so the cost is nearly flat in the number of
    starts. The runs remain statistically independent -- only their bookkeeping
    is shared.

    Two differences from [`_run_sequential`][] follow from batching:

    - `time_limit` is a single budget for the whole batch, checked once per
      iteration, so all runs stop at the same iteration.
    - Random draws are batched across runs, so a given seed does not reproduce
      the sequential path's draw order.

    Args:
        instance: See [`_run_sequential`][].
        starts: See [`_run_sequential`][].
        top_k: See [`_run_sequential`][].
        max_iter: See [`_run_sequential`][].
        initial_temp: See [`_run_sequential`][].
        alpha: See [`_run_sequential`][].
        time_limit: Wall-clock budget in seconds, for the whole batch.
        rng: See [`_run_sequential`][].
        stats: See [`solve`][].

    Returns:
        One [`Solution`][] per row of `starts`, in the same order, each
            sorted by ascending cost with probabilities computed.
    """
    Q = instance.matrix
    n = Q.shape[0]
    n_runs = starts.shape[0]

    X = starts.detach().clone().to(Q)
    QX = X @ Q
    energy = batched_quadratic_cost(X, Q)

    # Preallocated for the worst case (every iteration runs) and sliced down to
    # however many actually did; each iteration writes the post-move state of
    # every run at index `visits`, with index 0 holding the starting state.
    visited_bits = torch.empty((max_iter + 1, n_runs, n), dtype=bitstring.dtype(), device=X.device)
    visited_energy = torch.empty((max_iter + 1, n_runs), dtype=energy.dtype, device=X.device)
    visited_bits[0] = X.to(bitstring.dtype())
    visited_energy[0] = energy

    temperature: float = initial_temp
    rows = torch.arange(n_runs, device=X.device)
    deadline = time.perf_counter() + time_limit
    visits = 1

    for _ in range(max_iter):
        if time.perf_counter() >= deadline:
            break

        # `energy` and `QX` are accumulated incrementally, so each step adds a
        # rounding error that leaves the reported costs a few ULPs off the true
        # x^T Q x. Recomputing them exactly every _REFRESH_EVERY steps keeps
        # that error from becoming visible, at the cost of one extra matmul
        # amortized over many iterations. `X` itself never needs this: each
        # accepted flip moves it by an exact +1/-1, so it stays exact with no
        # drift to correct.
        if visits % _REFRESH_EVERY == 0:
            QX = X @ Q
            energy = batched_quadratic_cost(X, Q)

        idx = torch.randint(0, n, (n_runs,), generator=rng, device=X.device)
        dE = _flip_deltas(Q, X, QX).gather(1, idx.unsqueeze(1)).squeeze(1)

        accept = (dE <= 0.0) | (
            torch.rand(n_runs, generator=rng, device=X.device, dtype=X.dtype)
            < torch.exp(-dE / temperature)
        )

        # Flipping x -> 1 - x moves the bit by +1 or -1; zeroing that step on
        # the rejected runs leaves them untouched without branching per run.
        xi = X[rows, idx]
        step = (1.0 - 2.0 * xi) * accept.to(X.dtype)
        X[rows, idx] = xi + step
        energy = energy + step.abs() * dE
        QX = QX + step.unsqueeze(1) * Q[idx, :]

        visited_bits[visits] = X.to(bitstring.dtype())
        visited_energy[visits] = energy
        visits += 1

        temperature *= alpha
        if temperature < 1e-12:
            temperature = 1e-12

    # (visits, n_runs, n) and (visits, n_runs): one entry per recorded state.
    visited_bits = visited_bits[:visits]
    visited_energy = visited_energy[:visits]

    solutions: list[Solution] = []
    for r in range(n_runs):
        # Every recorded state starts as its own candidate counting a single
        # visit; `deduplicate` then collapses repeats of the same bitstring,
        # summing those visits into its count and keeping its energy. Since it
        # leaves the result sorted by ascending cost, `truncate` reduces it to
        # the top_k lowest-energy ones. This loop runs once per run rather than
        # once per iteration, so it stays off the hot path.
        solution = Solution(
            bitstrings=bitstrings.as_tensor(visited_bits[:, r, :]),
            costs=visited_energy[:, r],
            counts=vectori.zeros(visits).fill_(1),
            probabilities=vector.zeros(visits).fill_(1.0 / visits),
        )
        # `costs` was accumulated incrementally, drifting from the true x^T Q x
        # by up to _REFRESH_EVERY steps of rounding error. `deduplicate` picks
        # the row to keep per bitstring based on that drifted cost, so skip its
        # own recompute (`update=False`) and instead recompute exactly via
        # `_update` right after, before `truncate` selects on it.
        solution.deduplicate(update=False)._update(instance).truncate(top_k)

        if stats == "per_run":
            solution.counts = solution.counts.clone().fill_(1)
            solution._compute_probabilities()

        solutions.append(solution)

    return solutions
