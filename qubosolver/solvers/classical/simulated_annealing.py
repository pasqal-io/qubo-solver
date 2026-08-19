"""Simulated Annealing solver for QUBO problems.

Implements a bit-flip annealer that minimises the quadratic objective
$E(x) = x^T Q x$ over binary vectors $x \\in \\{0,1\\}^n$, run independently from
each of a batch of starting points and merged into a single solution.

The public entry point is `simulated_annealing`.  It is called by
:class:`~qubosolver.solvers.SimulatedAnnealingSolver`.
"""

from __future__ import annotations

import time
import heapq
import logging
from dataclasses import dataclass
from typing import Literal

import torch
from typing_extensions import overload

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


@overload
def simulated_annealing(
    instance: Instance,
    start: Bitstrings,
    *,
    merge: Literal[True] = True,
    top_k: int = 1,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    time_limit: float = float("inf"),
    rng: torch.Generator = _default_rng,
    stats: Literal["per_run", "full"] = "per_run",
) -> Solution: ...


@overload
def simulated_annealing(
    instance: Instance,
    start: Bitstrings,
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
) -> list[Solution]: ...


@torch.no_grad()
def simulated_annealing(
    instance: Instance,
    start: Bitstrings,
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
) -> Solution | list[Solution]:
    """Run Simulated Annealing on a QUBO instance from each of a batch of starting points.

    For each starting bitstring, at each of `max_iter` steps a random bit is
    proposed for flipping.  The flip is always accepted when it reduces the
    energy; otherwise it is accepted with probability $\\exp(-\\Delta E / T)$.
    Energy updates are computed incrementally in O(n) per step using the
    cached matrix-vector product ``Qx``.

    Up to `top_k` unique lowest-energy bitstrings encountered during each run
    are retained, along with how many iterations were spent at each one
    (whether or not the proposed flip at that iteration was accepted). The
    runs are independent.  By default (``merge=True``) the per-start results
    are merged into a single `Solution` via ``Solution.concat(...).deduplicate()``;
    pass ``merge=False`` to instead get back the unmerged, one-per-start list.

    Example:
        Running a single start requires promoting it to a batch of size 1
        first, via ``unsqueeze``:

        ```python
        solution = simulated_annealing(instance, start.unsqueeze(0))
        ```

        Passing ``merge=False`` returns the per-start results instead of a
        single merged `Solution`:

        ```python
        solutions = simulated_annealing(instance, start, merge=False)
        ```

    Args:
        instance: The QUBO instance to solve.  Its coefficient matrix is
            symmetrised internally as ``(Q + Qᵀ) / 2``.
        start: Batch of initial binary solutions, a tensor of shape
            ``(k, n)`` with values in ``{0, 1}``.  One independent run is
            performed per row.
        merge: When ``True`` (default), merge the per-start results into a
            single `Solution` via ``Solution.concat(...).deduplicate()``.
            When ``False``, return the unmerged list of one `Solution` per
            starting point (same order as `start`).
        top_k: Maximum number of unique best solutions to keep per run,
            ordered by ascending energy. Defaults to ``1``.
        max_iter: Number of bit-flip proposals to perform.
        initial_temp: Starting temperature $T_0$.  Higher values increase the
            probability of accepting uphill moves early in the search.
            Defaults to ``5.0``.
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
        rng: PyTorch random number generator used for bit selection and
            acceptance sampling.  Defaults to a module-level
            generator created once at import time; pass an explicit generator
            for reproducibility across calls.
        stats: When ``"per_run"`` (default), each run's retained bitstrings
            are counted as ``1`` instead of how many iterations were spent
            at each one, before any merging. This is mainly meant for
            ``top_k=1`` together with ``merge=True``, where the merged count
            directly reflects how many of the runs converged on each
            bitstring. With ``merge=True`` and ``top_k > 1``, or whenever a
            bitstring is retained by more than one run, `deduplicate` sums
            those per-run ``1``s, so counts on the merged result are
            generally neither ``1`` nor uniform. When ``"full"``, counts
            instead reflect how many iterations were spent at each
            bitstring.

    Returns:
        When ``merge=True``, a single [`Solution`][] merging every start's
        results.  When ``merge=False``, one [`Solution`][] per starting
        point (same order as `start`).  Either way, each `Solution`
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
    Q = instance.matrix

    # determine cooling rate alpha
    if max_iter <= 1:
        alpha = 1.0
    elif cooling_rate is not None:
        alpha = float(cooling_rate)
        if not (0.0 < alpha < 1.0):
            raise ValueError("cooling_rate (alpha) must be in (0, 1).")
    else:
        alpha = (final_temp / initial_temp) ** (1.0 / (max_iter - 1))

    solutions: list[Solution] = []

    for b in start:
        bits: Bitstring = b.detach().clone()

        Qx = Q @ bits.to(Q)
        energy = float(bits.to(Q).dot(Qx))

        temperature: float = initial_temp

        visited_solutions: dict[bytes, _Data] = {}
        visited_solutions[_to_key(bits)] = _Data(energy, 1)

        deadline = time.perf_counter() + time_limit

        for _ in range(max_iter):
            if time.perf_counter() >= deadline:
                break

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
            bitstrings=bitstrings.from_torch(unique_bits),
            costs=vector.tensor([s.energy for s in visited_solutions.values()]),
            counts=counts,
        )

        solutions.append(solution._sort_by_cost()._compute_probabilities())

    if merge:
        return Solution.concat(solutions).deduplicate()

    return solutions
