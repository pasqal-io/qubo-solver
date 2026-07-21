"""Simulated Annealing solver for QUBO problems.

Implements a single-run bit-flip annealer that minimises
the quadratic objective E(x) = xᵀ Q x over binary vectors x ∈ {0,1}ⁿ.

The public entry point is `simulated_annealing`.  It is called by
:class:`~qubosolver.solvers.SimulatedAnnealingSolver` and used as the first
phase of :class:`~qubosolver.solvers.HybridSATabuSolver`.
"""

from __future__ import annotations

import time

import torch

from qubosolver import Instance, Solution, bitstrings, vector, Bitstring, torch_rng


@torch.no_grad()
def simulated_annealing(
    qubo: Instance,
    start: Bitstring,
    *,
    top_k: int = 5,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    energy_tol: float = 0.0,
    time_limit: float = float("inf"),
    rng: torch.Generator = torch_rng(),
) -> Solution:
    """Run Simulated Annealing on a QUBO instance and return the best solutions found.

    At each of `max_iter` steps a random bit is proposed for flipping.  The
    flip is always accepted when it reduces the energy; otherwise it is
    accepted with probability ``exp(-ΔE / T)``.  Energy updates are computed
    incrementally in O(n) per step using the cached matrix-vector product ``Qx``.

    Up to `top_k` unique lowest-energy bitstrings encountered during the run
    are retained and returned, deduplicated by byte-hashing.

    Args:
        qubo: The QUBO instance to solve.  Its coefficient matrix is
            symmetrised internally as ``(Q + Qᵀ) / 2``.
        start: Initial binary solution tensor of shape ``(n,)`` with values
            in ``{0, 1}``.  The search begins from this configuration.
        top_k: Maximum number of unique best solutions to keep, ordered by
            ascending energy.
        max_iter: Number of bit-flip proposals to perform.
        initial_temp: Starting temperature T₀.  Higher values increase the
            probability of accepting uphill moves early in the search.
            Defaults to ``5.0``.
        final_temp: Target temperature T_f at the end of the schedule, used
            to derive the cooling rate when `cooling_rate` is ``None``.
            Ignored when `cooling_rate` is provided explicitly.
        cooling_rate: Geometric cooling factor α ∈ (0, 1) such that
            T ← α·T at each step.  When ``None`` (default), α is derived
            automatically from `initial_temp`, `final_temp`, and `max_iter`
            so that the temperature reaches `final_temp` after `max_iter` steps.
        energy_tol: Two solutions with energies differing by at most this
            value are treated as equivalent when maintaining the top-k list.
            Defaults to ``0.0`` (strict equality).
        time_limit: Wall-clock budget in seconds.  The algorithm stops early
            when either `max_iter` steps or the time limit is reached,
            whichever comes first.  Defaults to ``float("inf")`` (no limit).
        rng: PyTorch random number generator used for bit selection and
            acceptance sampling.  Defaults to a module-level
            generator created once at import time; pass an explicit generator
            for reproducibility across calls.

    Returns:
        A solution containing up to `top_k` unique bitstrings sorted by ascending energy, with their costs, counts (how many times each was visited during the run), and normalised probabilities.

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

    Q = qubo.matrix
    Q = 0.5 * (Q + Q.T)
    n = int(Q.shape[0])

    bits = start.to(device=Q.device, dtype=torch.uint8).clamp_(0, 1)

    bits_f = bits.to(dtype=Q.dtype)
    Qx = Q @ bits_f
    energy = float(bits_f.dot(Qx))

    # determine cooling rate alpha
    if max_iter <= 1:
        alpha = 1.0
    elif cooling_rate is not None:
        alpha = float(cooling_rate)
        if not (0.0 < alpha < 1.0):
            raise ValueError("cooling_rate (alpha) must be in (0, 1).")
    else:
        alpha = float((final_temp / initial_temp) ** (1.0 / (max_iter - 1)))

    temperature = float(initial_temp)

    top_sol: list[tuple[float, torch.Tensor]] = []
    seen: set[bytes] = set()

    def key_from_bits(b: torch.Tensor) -> bytes:
        return bytes(b.cpu().tolist())

    def maybe_insert(b_u8: torch.Tensor, e: float) -> None:
        k = key_from_bits(b_u8)
        if k in seen:
            return
        top_sol.append((e, b_u8.cpu().clone()))
        top_sol.sort(key=lambda t: t[0])
        if len(top_sol) > top_k:
            cutoff = top_sol[top_k - 1][0]
            kept = [pair for pair in top_sol if pair[0] <= cutoff + energy_tol]
            top_sol[:] = kept[:top_k]
        seen.clear()
        for _, bb in top_sol:
            seen.add(key_from_bits(bb))

    maybe_insert(bits, energy)

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
            torch.rand((), generator=rng).item() < torch.exp(torch.tensor(-dE / temperature)).item()
        )
        if accept:
            new_xi = 1 - xi
            diff = float(new_xi - xi)
            bits[i] = new_xi
            bits_f[i] = float(new_xi)
            energy += dE
            Qx += diff * Q[:, i]
            if len(top_sol) < top_k or energy <= top_sol[-1][0] + energy_tol:
                maybe_insert(bits, energy)

        temperature *= alpha
        if temperature < 1e-12:
            temperature = 1e-12

    top_sol.sort(key=lambda t: t[0])
    bitstrings_ = bitstrings.from_torch(torch.stack([b for (_, b) in top_sol], dim=0))
    energies = vector.tensor([e for (e, _) in top_sol])

    unique_bits, inverse_indices, counts = torch.unique(
        bitstrings_, dim=0, return_inverse=True, return_counts=True
    )
    energies = energies[inverse_indices]

    solution = Solution(
        bitstrings=unique_bits, counts=counts, costs=energies
    ).compute_probabilities()
    return solution
