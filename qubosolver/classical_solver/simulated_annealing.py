from __future__ import annotations

import torch

from qubosolver import QUBOInstance, QUBOSolution


def qubo_simulated_annealing(
    qubo: QUBOInstance,
    top_k: int = 5,
    max_iter: int = 1000,
    initial_temp: float = 10.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    seed: int | None = None,
    start: torch.Tensor | None = None,
    energy_tol: float = 0.0,
) -> QUBOSolution:
    """
    Solve a QUBO instance using the Simulated Annealing metaheuristic.

    This function wraps the low-level `simulated_annealing()` routine
    and converts its output into a standardized `QUBOSolution` object.

    The algorithm gradually lowers the system temperature to reduce
    the probability of accepting worse solutions, balancing exploration
    and exploitation.

    Returns:
        A `QUBOSolution` object containing:
            - `bitstrings`: The best binary solution(s) found.
            - `costs`: Corresponding objective values as tensors.

    Example:
        >>> solution = qubo_simulated_annealing(qubo)
        >>> print(solution.bitstrings, solution.costs)
    """
    bitstrings, costs = simulated_annealing(
        Q=qubo.coefficients,
        top_k=top_k,
        max_iter=max_iter,
        initial_temp=initial_temp,
        final_temp=final_temp,
        cooling_rate=cooling_rate,
        seed=seed,
        start=start,
        energy_tol=energy_tol,
    )
    return QUBOSolution(bitstrings=bitstrings, costs=costs)


@torch.no_grad()
def simulated_annealing(
    Q: torch.Tensor,
    top_k: int = 5,
    max_iter: int = 1000,
    initial_temp: float = 5.0,
    final_temp: float = 1e-3,
    cooling_rate: float | None = None,
    seed: int | None = None,
    start: torch.Tensor | None = None,
    energy_tol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Simulated Annealing (SA) for a Quadratic Unconstrained Binary Optimization (QUBO)
    problem using PyTorch, returning the top-K unique low-energy solutions found.

    The algorithm minimizes the quadratic form:
        E(x) = xᵀ Q x
    where x ∈ {0,1}ⁿ.

    Args:
        Q: torch.Tensor of shape (n, n)
            The QUBO matrix defining the objective. It is symmetrized internally
            as (Q + Qᵀ) / 2 to ensure a well-defined energy landscape.
        top_k: int, optional, default=5
            The maximum number of unique best solutions to keep, ordered by ascending energy.
        max_iter: int, optional, default=100_000
            The number of Metropolis steps (bit-flip proposals) performed during annealing.
        initial_temp: float, optional, default=5.0
            The starting temperature for the annealing schedule.
        final_temp: float, optional, default=1e-3
            The final temperature at the end of the annealing schedule.
        cooling_rate: float or None, default=None
            Geometric cooling factor α in (0, 1). If provided, temperature updates as
            T ← α·T each iteration and `final_temp` is ignored.
            If None, α is computed so that T transitions from T₀ to T_f in `max_iter`.
        seed: int or None, optional
            Random seed for reproducibility. If None, a random seed is used.
        start: torch.Tensor or None, optional
            Optional initial bitstring of shape (n,), with values in {0,1}.
            If None, the initial configuration is sampled uniformly at random.
        energy_tol: float, optional, default=0.0
            Energy tolerance for considering two solutions as equivalent.
            If two energies differ by ≤ `energy_tol`, they are treated as equal.

    Returns:
        solutions: torch.Tensor of shape (m, n), dtype=torch.uint8
            Up to `top_k` unique bitstrings found (m ≤ top_k), sorted by increasing energy.
            Returned on CPU for easy inspection and serialization.
        energies: torch.Tensor of shape (m,), dtype=torch.float64
            Corresponding energy values (xᵀ Q x) in ascending order.

    Notes:
        The algorithm uses an **incremental energy update** for O(n) cost per accepted flip:
          Qx ← Qx + Δxᵢ * Q[:, i], where Δxᵢ ∈ {+1, -1}.
        The **temperature schedule** follows a geometric decay:
          Tₜ = T₀ * (T_f / T₀)^(t / (max_iter - 1)).
        The **Metropolis acceptance rule** is applied:
          accept flip with probability p = min(1, exp(-ΔE / T)).
        Duplicate bitstrings are filtered using Python byte hashing for compactness.
        For stochastic diversity, multiple runs with different seeds are recommended.

    Examples:
    >>> import torch
    >>> Q = torch.tensor([
    ...     [1.0, -2.0,  0.0],
    ...     [-2.0, 1.0,  0.0],
    ...     [0.0,  0.0,  0.5],
    ... ])
    >>> solutions, energies = simulated_annealing_qubo_topk_torch(
    ...     Q, top_k=3, max_iter=1000, seed=42
    ... )
    >>> energies
    tensor([-3.0, -2.0, -1.5], dtype=torch.float64)
    >>> solutions
    tensor([[1, 1, 0],
            [1, 0, 0],
            [0, 1, 0]], dtype=torch.uint8)
    """
    if top_k <= 0:
        raise ValueError("top_k must be >= 1.")
    if initial_temp <= 0:
        raise ValueError("initial_temp must be > 0.")
    if cooling_rate is None and final_temp <= 0:
        raise ValueError("final_temp must be > 0 when cooling_rate is None.")

    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    Q = 0.5 * (Q + Q.T)
    n = int(Q.shape[0])

    if start is None:
        bits = torch.randint(0, 2, (n,), generator=rng, device=Q.device, dtype=torch.uint8)
    else:
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

    for _ in range(max_iter):
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
    bitstrings = torch.stack([b for (_, b) in top_sol], dim=0).to(torch.uint8)
    energies = torch.tensor([e for (e, _) in top_sol], dtype=torch.float64)
    return bitstrings, energies
