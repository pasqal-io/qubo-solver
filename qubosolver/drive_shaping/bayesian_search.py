"""Bayesian-optimised drive schedule generation for QUBO solving."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from skopt import gp_minimize
from collections.abc import Callable, Sequence
from typing import TypedDict
import torch

import qoolqit

from qubosolver.types import (
    Instance,
    Solution,
    Bitstring,
    Matrix,
    protocols,
    tensor,
)
from qubosolver import solvers, utils
from ._device_specs import max_virtual_amplitude, detuning_amplitude_ratio
from ._waveforms import constant_weighted_dmm


def _default_objective(solution: Solution) -> float:
    """Return the lowest cost from a solution, or infinity if empty."""
    return solution.costs[0].item() if solution else float("inf")


class _CallbackObjectiveInput(TypedDict):
    """Input dictionary passed to the optimisation callback."""

    x: Sequence[float]
    cost_eval: float


@dataclass
class Config:
    """Configuration for the Bayesian-optimisation drive shaper.

    Attributes:
        x0: Initial guess for the waveform parameters (3 amplitude + 3 detuning).
        n_calls: Number of Bayesian optimisation evaluations.
        seed: Random seed for reproducibility.
        qubo_cost: Callable used to evaluate bitstring cost against the QUBO matrix.
        objective: Callable that maps a `Solution` to a scalar
            objective (lower is better).
        callback_objective: Optional callback invoked after each evaluation.
        default_sequence_duration: Fallback maximum sequence duration (ns)
            injected when the target device has no `max_duration` cap.
    """

    x0: list[float] = field(
        default_factory=lambda: [
            0.5,
            0.9,
            0.5,
            -0.8,
            0.0,
            0.8,
        ]
    )
    n_calls: int = 20
    seed: int | None = None
    qubo_cost: Callable[[Bitstring, Matrix], float] = utils._costs.quadratic_cost
    objective: Callable[[Solution], float] = _default_objective
    callback_objective: Callable[[_CallbackObjectiveInput], None] = lambda data: None
    default_sequence_duration: int = 50000

    @staticmethod
    def from_drive_shaping_config(config: DriveShapingConfig) -> Config:
        """Create a `Config` from a user-facing [`DriveShapingConfig`][].

        Args:
            config: The drive-shaping configuration to convert.

        Returns:
            A configuration populated from the drive-shaping settings.
        """
        cfg = Config()
        cfg.x0 = (
            config.bayesian_search_initial_omega_parameters
            + config.bayesian_search_initial_detuning_parameters
        )
        cfg.n_calls = config.bayesian_search_n_calls
        cfg.seed = config.bayesian_search_seed
        cfg.default_sequence_duration = config.default_sequence_duration
        if config.bayesian_search_custom_qubo_cost is not None:
            cfg.qubo_cost = config.bayesian_search_custom_qubo_cost
        if config.bayesian_search_custom_objective is not None:
            cfg.objective = config.bayesian_search_custom_objective
        if config.bayesian_search_callback_objective is not None:
            cfg.callback_objective = config.bayesian_search_callback_objective

        return cfg


def _compute_norm_weights(instance: Instance) -> list[float]:
    """Compute per-qubit normalised weights from the diagonal of the QUBO matrix.

    Each weight is defined as ``1 - |Q_ii| / max_j(|Q_jj|)``, so a qubit
    whose diagonal coefficient equals the maximum gets weight 0 (fully
    penalised) and a qubit with a zero diagonal coefficient gets weight 1
    (unrestricted).  These weights are passed to the
    :class:`~qoolqit.drive.DetuningMapModulator` to modulate the local
    detuning per qubit.

    Args:
        instance: The QUBO instance whose diagonal entries are used.

    Returns:
        A list of floats in ``[0, 1]``, one per qubit, representing the
        normalised DMM weights.  Returns all-zeros when every diagonal
        entry is zero.
    """
    weights_list = torch.abs(torch.diag(instance.matrix)).tolist()
    max_node_weight = max(weights_list) if weights_list else 1.0
    norm_weights_list = [
        (1 - (w / max_node_weight)) if max_node_weight != 0 else 0.0 for w in weights_list
    ]
    return norm_weights_list


def _build_drive(
    instance: Instance,
    params: Sequence[float],
    *,
    dmm: bool,
    device: qoolqit.Device,
    register: qoolqit.Register,
) -> qoolqit.Drive:
    """Build a :class:`~qoolqit.Drive` from a flat parameter vector.

    The first three values in *params* control the amplitude waveform and the
    remaining three control the detuning waveform.  Both are represented as
    :class:`~qoolqit.InterpolatedWaveform` waveforms over the full sequence duration.
    Raw parameters are normalised in ``[0, 1]`` or ``[-1, 1]`` and are scaled
    so that the amplitude waveform stays within what the compiler can realize
    on `device` for `register`, and the detuning waveform stays within the
    detuning budget available for that (realized) amplitude.

    When *dmm* is enabled **and** the final detuning value is positive, a
    :class:`~qoolqit.drive.DetuningMapModulator` is added with
    per-qubit weights derived from the diagonal of the QUBO matrix (see
    `_compute_norm_weights`).

    Args:
        instance: The QUBO instance, used to compute DMM weights when *dmm* is
            ``True``.
        params: Flat sequence of 6 normalised parameters —
            ``params[:3]`` are the three interior amplitude knots and
            ``params[3:]`` are the three detuning knots.  Both ends of the
            amplitude waveform are pinned to zero.
        dmm: If ``True``, attach a constant weighted
            :class:`~qoolqit.drive.DetuningMapModulator` when the final
            detuning is positive.
        device: Target quantum device.
        register: The physical register the drive will run on.

    Returns:
        A fully configured :class:`~qoolqit.Drive` ready for simulation.
    """
    max_seq_duration: float = device.specs["max_duration"] or 1e3
    max_amplitude = max_virtual_amplitude(device, register)

    amp_params = [1e-9] + list(params[:3]) + [1e-9]
    amp_params = [p * max_amplitude for p in amp_params]
    amp_wave = qoolqit.InterpolatedWaveform(max_seq_duration, amp_params)

    # QoolQit rescales only based on the amplitude, so the maximum
    # of the detuning depends on the amplitude.
    det_ratio = detuning_amplitude_ratio(device)
    det_scale = det_ratio * float(amp_wave.max()) * (1.0 - 1e-3)
    # FIXME: det_params of length 4 ? with last param as final det for dmm?
    det_params = [p * det_scale for p in params[3:]]
    det_wave = qoolqit.InterpolatedWaveform(max_seq_duration, det_params)

    wdetunings = None
    final_detuning = det_params[-1]
    if dmm and final_detuning > 0:
        energy_scale = device._target_amp / float(amp_wave.max())
        wdetunings = constant_weighted_dmm(
            _compute_norm_weights(instance),
            max_seq_duration,
            final_detuning=-final_detuning,
            device=device,
            energy_scale=energy_scale,
        )

    shaped_drive = qoolqit.Drive(amplitude=amp_wave, detuning=det_wave, dmm=wdetunings)

    return shaped_drive


def _run_simulation(
    Q: torch.Tensor,
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    device: qoolqit.Device,
    backend: protocols.Backend,
    config: Config,
) -> Solution:
    """Execute one quantum simulation and return a costed, sorted solution.

    Submits an analog quantum sampling job via
    `~qubosolver.solvers.analog_quantum_sampling` using the
    ``MAX_ENERGY`` compiler profile (the default), evaluates the QUBO cost
    for every returned bitstring with ``config.qubo_cost``, then sorts
    results by cost and computes sampling probabilities in-place.

    If the simulation or post-processing raises any exception the error is
    printed and an empty :class:`Solution` is returned, so callers must
    treat an empty solution as a failure signal.

    Args:
        Q: The raw QUBO coefficient matrix (``torch.Tensor``).
        register: Physical atom register describing qubit positions.
        drive: The drive sequence to apply during the simulation.
        device: Target quantum device that defines hardware constraints.
        backend: Execution backend used to run the quantum program.
        config: Optimisation configuration supplying the ``qubo_cost``
            callable used to evaluate each returned bitstring and the
            fallback sequence duration for devices without a native cap.

    Returns:
        A :class:`Solution` with ``costs``, ``bitstrings``,
        ``probabilities``, and ``counts`` populated and sorted by ascending
        cost.  Returns an empty :class:`Solution` on any failure.
    """
    try:
        job = solvers.analog_quantum_sampling(
            register,
            drive,
            backend,
            device,
            default_sequence_duration=config.default_sequence_duration,
        )
        solution = Solution.from_results(job.results())
        costs = [config.qubo_cost(b, Q) for b in solution.bitstrings]
        solution.costs = tensor.tensor(costs)
        solution._sort_by_cost()._compute_probabilities()
        return solution
    except Exception as e:
        print(f"Simulation failed: {e}")
        return Solution()


def build_drive(
    instance: Instance,
    register: qoolqit.Register,
    *,
    backend: protocols.Backend,
    device: qoolqit.Device,
    dmm: bool = False,
    config: Config = Config(),
) -> tuple[qoolqit.Drive, Solution]:
    """Generate a drive schedule via Bayesian optimisation.

    Uses ``skopt.gp_minimize`` to search over waveform parameters,
    running quantum simulations at each evaluation to minimise the
    QUBO cost.

    Args:
        instance: The QUBO instance to solve.
        register: The physical atom register.
        backend: Execution backend for running simulations during optimisation.
        device: Target quantum device.
        dmm: Whether to use the Detuning Map Modulator.
        config: Optimisation parameters (initial guess, number of calls, etc.).

    Returns:
        A tuple of the best `qoolqit.Drive` found and the corresponding `Solution`.
    """
    n_amp = 3
    n_det = 3

    eps = 0.0001
    zero = eps
    one = 1.0 - eps

    bounds = [(zero, one)] * n_amp + [(-one, -zero)] + [(-one, one)] * (n_det - 2) + [(zero, one)]

    def run(x: list[float], eval: bool = True) -> tuple[float, Solution, qoolqit.Drive]:

        solution = Solution()
        drive = _build_drive(
            instance,
            x,
            dmm=dmm,
            device=device,
            register=register,
        )

        try:
            solution = _run_simulation(
                instance.matrix,
                register,
                drive,
                device,
                backend,
                config,
            )
            if eval:
                cost_eval = config.objective(solution)
                if not np.isfinite(cost_eval):
                    print(f"[Warning] Non-finite cost encountered: {cost_eval} at x={x}")
                    cost_eval = 1e4
            else:
                cost_eval = float("nan")

        except Exception as e:
            print(f"[Exception] Error during simulation at x={x}: {e}")
            cost_eval = 1e4
        return cost_eval, solution, drive

    def objective(x: list[float]) -> float:
        cost_eval, _, _ = run(x)
        config.callback_objective({"x": x, "cost_eval": cost_eval})

        return cost_eval

    opt_result = gp_minimize(
        objective,
        bounds,
        x0=config.x0,
        n_calls=config.n_calls,
        random_state=config.seed,
    )

    best_params = opt_result.x if opt_result else config.x0
    _, solution, drive = run(best_params, eval=False)

    return drive, solution
