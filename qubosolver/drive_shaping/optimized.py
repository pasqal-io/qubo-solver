from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from skopt import gp_minimize
from collections.abc import Callable, Sequence
from typing import TypedDict
import torch

import qoolqit
from qoolqit.execution.compilation_functions import CompilerProfile

from qubosolver.types import (
    QUBOInstance,
    QUBOSolution,
    Bitstring,
    Matrix,
    _protocols,
    tensor,
    Labelling,
)
from qubosolver.config import DriveShapingConfig
from qubosolver import solvers, _utils
from ._waveforms import constant_weighted_dmm


def _default_objective(solution: QUBOSolution) -> float:
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
        objective: Callable that maps a :class:`QUBOSolution` to a scalar
            objective (lower is better).
        callback_objective: Optional callback invoked after each evaluation.
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
    qubo_cost: Callable[[Bitstring, Matrix], float] = _utils.costs.quadratic_cost
    objective: Callable[[QUBOSolution], float] = _default_objective
    callback_objective: Callable[[_CallbackObjectiveInput], None] = lambda data: None

    @staticmethod
    def from_drive_shaping_config(config: DriveShapingConfig) -> Config:
        """Create a :class:`Config` from a user-facing :class:`DriveShapingConfig`.

        Args:
            config: The drive-shaping configuration to convert.

        Returns:
            A :class:`Config` populated from the drive-shaping settings.
        """
        cfg = Config()
        cfg.x0 = (
            config.optimized_initial_omega_parameters + config.optimized_initial_detuning_parameters
        )
        cfg.n_calls = config.optimized_n_calls
        cfg.seed = config.optimized_seed
        if config.optimized_custom_qubo_cost is not None:
            cfg.qubo_cost = config.optimized_custom_qubo_cost
        if config.optimized_custom_objective is not None:
            cfg.objective = config.optimized_custom_objective
        if config.optimized_callback_objective is not None:
            cfg.callback_objective = config.optimized_callback_objective

        return cfg


def _compute_norm_weights(Q: QUBOInstance) -> list[float]:
    """Compute normalization weights.

    Returns:
        list[float]: normalization weights.
    """
    weights_list = torch.abs(torch.diag(Q.matrix)).tolist()
    max_node_weight = max(weights_list) if weights_list else 1.0
    norm_weights_list = [
        (1 - (w / max_node_weight)) if max_node_weight != 0 else 0.0 for w in weights_list
    ]
    return norm_weights_list


def _build_drive(
    Q: QUBOInstance,
    params: Sequence[float],
    *,
    dmm: bool,
    device_specs: dict[str, float | None],
    labelling: Labelling,
) -> qoolqit.Drive:
    """Build the drive from a list of parameters for the objective.

    Args:
        params (list): List of parameters.

    Returns:
        Drive: Drive sequence.
    """
    max_seq_duration: float = device_specs["max_duration"] or 1e3
    max_amplitude: float = device_specs["max_amplitude"] or 1e4
    max_detuning: float = device_specs["max_abs_detuning"] or 1e4

    amp_params = [1e-9] + list(params[:3]) + [1e-9]
    # FIXME: det_params of length 4 ? with last param as final det for dmm?
    det_params = list(params[3:])
    amp_params = [p * max_amplitude for p in amp_params]
    det_params = [p * max_detuning for p in det_params]

    amp_wave = qoolqit.Interpolated(max_seq_duration, amp_params)
    det_wave = qoolqit.Interpolated(max_seq_duration, det_params)

    wdetunings = None
    final_detuning = det_params[-1]
    if dmm and final_detuning > 0:
        wdetunings = constant_weighted_dmm(
            _compute_norm_weights(Q),
            max_seq_duration,
            final_detuning=-final_detuning,
            labelling=labelling,
        )

    shaped_drive = qoolqit.Drive(amplitude=amp_wave, detuning=det_wave, dmm=wdetunings)

    return shaped_drive


def _run_simulation(
    Q: torch.Tensor,
    register: qoolqit.Register,
    drive: qoolqit.Drive,
    device: qoolqit.Device,
    backend: _protocols.Backend,
    config: Config,
) -> QUBOSolution:
    """Run a quantum program using backend and returns
        a tuple of (bitstrings, counts, probabilities, costs, best cost, best bitstring).

    Args:
        register (Register): register of quantum program.
        drive (Drive): drive to run on backend.
        QUBO (torch.Tensor): Qubo coefficients.
        convert_to_tensor (bool, optional): Convert tuple components to tensors.
            Defaults to True.

    Returns:
        tuple: tuple of (bitstrings, counts, probabilities, costs, best cost, best bitstring)
    """
    try:
        job = solvers.analog_quantum_sample(
            register, drive, backend, device, compiler_profile=CompilerProfile.WORKING_POINT
        )
        solution = QUBOSolution.from_results(job.results())
        costs = [config.qubo_cost(b, Q) for b in solution.bitstrings]
        solution.costs = tensor.tensor(costs)
        solution.sort_by_cost().compute_probabilities()
        return solution
    except Exception as e:
        print(f"Simulation failed: {e}")
        return QUBOSolution()


def build_drive(
    Q: QUBOInstance,
    register: qoolqit.Register,
    backend: _protocols.Backend,
    device: qoolqit.Device,
    *,
    dmm: bool = False,
    config: Config = Config(),
    labelling: Labelling = str,
) -> tuple[qoolqit.Drive, QUBOSolution]:
    """Generate an optimised drive schedule via Bayesian optimisation.

    Uses ``skopt.gp_minimize`` to search over waveform parameters,
    running quantum simulations at each evaluation to minimise the
    QUBO cost.

    Args:
        register: The physical atom register.
        Q: The QUBO instance to solve.
        device: Target quantum device.
        dmm: Whether to use the Detuning Map Modulator.
        backend: Execution backend for running simulations during optimisation.
        config: Optimisation parameters (initial guess, number of calls, etc.).

    Returns:
        A tuple of the best :class:`~qoolqit.Drive` found and the
        corresponding :class:`QUBOSolution`.
    """
    n_amp = 3
    n_det = 3

    eps = 0.0001
    zero = eps
    one = 1.0 - eps

    bounds = [(zero, one)] * n_amp + [(-one, -zero)] + [(-one, one)] * (n_det - 2) + [(zero, one)]

    def run(x: list[float], eval: bool = True) -> tuple[float, QUBOSolution, qoolqit.Drive]:

        solution = QUBOSolution()
        drive = _build_drive(
            Q,
            x,
            dmm=dmm,
            device_specs=device.specs,
            labelling=labelling,
        )

        try:
            solution = _run_simulation(
                Q.matrix,
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
