from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
from skopt import gp_minimize

from pulser.devices import AnalogDevice
from qoolqit import Register, QuantumProgram, Drive
from qoolqit.waveforms import Interpolated as InterpolatedWaveform
from qubosolver import concepts


from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import DriveType
from qubosolver.utils import calculate_qubo_cost
from qubosolver.pipeline.waveforms import weighted_detunings


class BaseDriveShaper(ABC):
    """
    Abstract base class for generating Qoolqit drives based on a QUBO problem.

    This class transforms the structure of a QUBOInstance into a quantum
    waveform sequence or drive that can be applied to a physical register. The register
    is passed at the time of drive generation, not during initialization.

    Attributes:
        instance (QUBOInstance): The QUBO problem instance.
        config (SolverConfig): The solver configuration.
        drive (Drive, optional): A saved current drive obtained by `generate`.
        backend (Backend): Backend to use.
        device (Device): Device from backend.

    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: concepts.Backend):
        """
        Initialize the drive shaping module with a QUBO instance.

        Args:
            instance (QUBOInstance): The QUBO problem instance.
            config (SolverConfig): The solver configuration.
            backend (Backend): Backend to use.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.drive: Drive | None = None
        self.backend = backend
        self.device = self.config.device

        # check if device allow DMM
        self.dmm = self.config.drive_shaping.dmm and (
            len(list(self.config.device._device.dmm_channels.keys())) > 0
        )

    @property
    def qubo_coefficients(self) -> torch.Tensor:
        return self.instance.coefficients

    def _compute_norm_weights(self) -> list[float]:
        """Compute normalization weights.

        Returns:
            list[float]: normalization weights.
        """
        TIME, _, _ = self.device.converter.factors
        weights_list = torch.abs(torch.diag(self.qubo_coefficients)).tolist()
        max_node_weight = max(weights_list) if weights_list else 1.0
        norm_weights_list = [
            (1 - (w / max_node_weight)) / TIME if max_node_weight != 0 else 0.0
            for w in weights_list
        ]
        return norm_weights_list

    def _scale_omega_for_device_constraints(self, parameter: float) -> float:
        """Scale the parameter given the device `min_avg_amp` and `max_amp` constraints."""
        rydberg_global = self.device._device.channels["rydberg_global"]
        min_avg_amp = rydberg_global.min_avg_amp
        max_amp = rydberg_global.max_amp

        if min_avg_amp is not None:
            parameter = max(parameter, min_avg_amp + 1e-9)
        if max_amp is not None:
            parameter = min(parameter, max_amp - 1e-9)

        return parameter

    @abstractmethod
    def generate(
        self,
        register: Register,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate a drive based on the problem and the provided register.

        Args:
            register (Register): The physical register layout.

        Returns:
            Drive: A generated Drive.
            QUBOSolution: An instance of the qubo solution
        """
        pass


class AdiabaticDriveShaper(BaseDriveShaper):
    """
    A Standard Adiabatic Drive shaper.
    """

    def _find_max_interaction_coeff_vectorized(self) -> float:
        """
        Finds the maximum q_ij such that q_ii + q_jj + q_ij + q_ji < 0,
        using vectorized operations.

        Returns:
            float: The maximum q_ij value found, inf if no value satisfies
                the confition.
        """
        Q = self.qubo_coefficients
        n = Q.shape[0]
        i_indices, j_indices = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        q_ii = Q[i_indices, i_indices]
        q_jj = Q[j_indices, j_indices]

        q_ij = Q[i_indices, j_indices]
        q_ji = Q[j_indices, i_indices]

        condition_mask = (q_ii + q_jj + q_ij + q_ji) < 0
        valid_q_ij_values = Q[condition_mask]
        if valid_q_ij_values.numel() == 0:
            return float("inf")

        return float(torch.max(valid_q_ij_values).cpu().item())

    def generate(
        self,
        register: Register,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate an adiabatic drive based on the QUBO instance and physical register.

        Args:
            register (Register): The physical register layout for the quantum system.

        Returns:
            tuple[Drive, QUBOSolution | None]:
                - Drive: A generated Drive object.
                - QUBOSolution: An instance of the qubo solution
                    - str | None: The bitstring (solution) -> Not computed
                    - float | None: The cost (energy value) -> Not computed
                    - float | None: The probabilities for each bitstring -> Not computed
                    - float | None: The counts of each bitstring -> Not computed
        """

        # for conversions to qoolqit
        TIME, ENERGY, _ = self.device.converter.factors
        QUBO = self.qubo_coefficients

        norm_weights_list = self._compute_norm_weights()

        off_diag = QUBO[
            ~torch.eye(QUBO.shape[0], dtype=torch.bool)
        ]  # Selecting off-diagonal terms of the Qubo with a mask

        # device constraints
        rydberg_global = self.device._device.channels["rydberg_global"]
        min_avg_amp = rydberg_global.min_avg_amp
        max_amp = rydberg_global.max_amp

        Omega = torch.mean(off_diag).item()
        sign = 1.0 if Omega >= 0 else -1.0
        mag = abs(Omega)
        if min_avg_amp:
            # to make the average values higher then the minimum
            # use the average value of a parabola for
            # the amplitude waveform with Omega
            mag = max(mag, ENERGY * (3.0 * (min_avg_amp + 1e-9) / 2.0))
        if max_amp:
            mag = min(
                mag,
                max_amp - 1e-9,
            )
        Omega = sign * mag

        delta_0 = torch.min(torch.diag(QUBO)).item()
        delta_f = -delta_0

        # enforces AnalogDevice max sequence duration if device has no max
        max_seq_duration_ = (
            self.device._device.max_sequence_duration or AnalogDevice.max_sequence_duration
        )
        assert max_seq_duration_ is not None

        max_seq_duration = max_seq_duration_ / TIME
        Omega /= TIME
        delta_0 /= TIME
        delta_f /= TIME

        amp_wave = InterpolatedWaveform(max_seq_duration, [1e-9 / TIME, Omega, 1e-9 / TIME])
        det_wave = InterpolatedWaveform(max_seq_duration, [delta_0, 0, delta_f])

        wdetunings = None
        if self.dmm and delta_f > 0:
            wdetunings = weighted_detunings(
                register,
                max_seq_duration,
                norm_weights_list,
                -delta_f,
            )

        shaped_drive = Drive(amplitude=amp_wave, detuning=det_wave, weighted_detunings=wdetunings)
        solution = QUBOSolution(torch.Tensor(), torch.Tensor())

        return shaped_drive, solution

class HeuristicDriveShaper(BaseDriveShaper):
    """
    Heuristic schedule drive shaper.

    Key idea:
      - Encode QUBO diagonals exactly at final time using:
            d_i = -alpha * Q_ii
        but respecting the existing DMM convention used in this repo:
        DMM contributes as a NEGATIVE detuning map (it "pulls down" detuning locally).

      - We therefore set:
            delta_g(T)   = d_max
            delta_dmm(T) = -(d_max - d_min) <= 0
            w_i          = (d_max - d_i) / (d_max - d_min) in [0, 1]
        so that:
            delta_i(T) = delta_g(T) + delta_dmm(T) * w_i = d_i
    """

    @staticmethod
    def _clip(x: float, lo: float | None, hi: float | None) -> float:
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x

    def _scale_omega_for_device_constraints(self, omega_wave_units: float) -> float:
        """
        Ensure omega respects Pulser channel constraints in *waveform units*
        (i.e., the units used inside InterpolatedWaveform samples after division by TIME).
        """
        rydberg_global = self.device._device.channels["rydberg_global"]
        min_avg_amp = rydberg_global.min_avg_amp
        max_amp = rydberg_global.max_amp

        # Safety epsilon to avoid boundary rounding issues in Pulser validation
        eps = 1e-9

        # Clamp to max_amp (peak constraint)
        if max_amp is not None:
            omega_wave_units = min(omega_wave_units, float(max_amp) - eps)

        # Ensure not trivially below min_avg_amp if defined (best-effort).
        # NOTE: min_avg_amp is an average constraint; we approximate by ensuring the plateau is above it.
        if min_avg_amp is not None:
            omega_wave_units = max(omega_wave_units, float(min_avg_amp) + eps)

            # Re-apply max after enforcing min
            if max_amp is not None:
                omega_wave_units = min(omega_wave_units, float(max_amp) - eps)

        # Avoid negative or zero
        omega_wave_units = max(omega_wave_units, eps)
        return omega_wave_units

    def _get_hw_detuning_bound(self) -> float | None:
        ch = self.device._device.channels["rydberg_global"]
        return ch.max_abs_detuning

    def _get_hw_dmm_bound(self) -> float | None:
        # DMM channels (if present) — best-effort
        dmm_channels = list(getattr(self.device._device, "dmm_channels", {}).values())
        if not dmm_channels:
            return None
        return getattr(dmm_channels[0], "max_abs_detuning", None)

    def _compute_alpha_diag_max(
        self,
        qmin: float,
        qmax: float,
        delta_g_min: float,
        delta_g_max: float,
        delta_dmm_max: float,
    ) -> float:
        """
        Compute a conservative alpha_max so that final encoding fits hardware.

        With our convention:
          d_i = -alpha * Q_ii
          delta_g(T)   = d_max = max_i d_i = -alpha * qmin
          delta_dmm(T) = -(d_max - d_min) = -alpha*(qmax - qmin)  (<= 0)
          |delta_dmm(T)| <= delta_dmm_max
          delta_g(T) within [delta_g_min, delta_g_max]
        """
        # If diagonal is constant, range is 0 -> no DMM amplitude needed
        if abs(qmax - qmin) < 1e-15:
            # Only need delta_g(T) = -alpha*qmin within global bounds.
            if abs(qmin) < 1e-15:
                return 1.0  # anything works, pick 1
            # Solve: delta_g_min <= -alpha*qmin <= delta_g_max
            # We want alpha > 0
            candidates = []
            if qmin > 0:
                # -alpha*qmin <= delta_g_max  => alpha >= -delta_g_max/qmin (but delta_g_max may be positive)
                # and -alpha*qmin >= delta_g_min => alpha <= -delta_g_min/qmin
                candidates.append((-delta_g_min) / qmin)
            else:
                # qmin < 0: -alpha*qmin is positive
                candidates.append(delta_g_max / (-qmin))
            return max(0.0, min(candidates)) if candidates else 0.0

        # DMM magnitude constraint:
        # |delta_dmm(T)| = alpha*(qmax - qmin) <= delta_dmm_max
        alpha_dmm = delta_dmm_max / (qmax - qmin)

        # Global final detuning constraint:
        # delta_g(T) = d_max = -alpha*qmin must be within [delta_g_min, delta_g_max]
        alpha_global = float("inf")
        if abs(qmin) > 1e-15:
            if qmin < 0:
                # -alpha*qmin is positive, enforce <= delta_g_max
                alpha_global = delta_g_max / (-qmin)
            else:
                # -alpha*qmin is negative, enforce >= delta_g_min
                alpha_global = (-delta_g_min) / qmin

        return float(max(0.0, min(alpha_dmm, alpha_global)))

    def generate(self, register: Register) -> tuple[Drive, QUBOSolution]:
        # Conversions
        TIME, ENERGY, _ = self.device.converter.factors
        Q = self.qubo_coefficients

        # Sequence duration
        max_seq_duration = (
            self.device._device.max_sequence_duration or AnalogDevice.max_sequence_duration
        )
        assert max_seq_duration is not None
        max_seq_duration = max_seq_duration / TIME  # waveform time units

        # Hardware detuning bounds (global)
        max_abs_det = self._get_hw_detuning_bound()
        if max_abs_det is None:
            max_abs_det = 1e6  # fallback
        delta_g_min = -float(max_abs_det)
        delta_g_max = float(max_abs_det)

        # Hardware DMM bounds (magnitude). If no DMM -> 0
        if self.dmm:
            max_abs_dmm = self._get_hw_dmm_bound()
            if max_abs_dmm is None:
                max_abs_dmm = max_abs_det
            delta_dmm_max = float(max_abs_dmm)
        else:
            delta_dmm_max = 0.0

        # Diagonal stats (NO /TIME here)
        diag = torch.diag(Q)
        if diag.numel() == 0:
            # trivial
            eps = 1e-9 / TIME
            amp_wave = InterpolatedWaveform(max_seq_duration, [eps, eps])
            det_wave = InterpolatedWaveform(max_seq_duration, [0.0, 0.0])
            return Drive(amplitude=amp_wave, detuning=det_wave, weighted_detunings=None), QUBOSolution(
                torch.Tensor(), torch.Tensor()
            )

        qmin = float(torch.min(diag).cpu().item())
        qmax = float(torch.max(diag).cpu().item())

        # Choose alpha with margin
        alpha_max = self._compute_alpha_diag_max(
            qmin=qmin,
            qmax=qmax,
            delta_g_min=delta_g_min,
            delta_g_max=delta_g_max,
            delta_dmm_max=delta_dmm_max,
        )
        safety = float(getattr(self.config.drive_shaping, "heuristic_alpha_safety", 0.8))
        alpha = safety * alpha_max if alpha_max > 0 else 0.0
        if alpha <= 0:
            # last resort (keeps pipeline alive, but won't encode well)
            alpha = 1e-6

        # Target per-site final detunings (energy units)
        d_i = (-alpha * diag).cpu().numpy()
        d_min = float(np.min(d_i))
        d_max = float(np.max(d_i))
        spread = max(0.0, d_max - d_min)

        # Global final detuning is the TOP value; DMM pulls down
        delta_g_T = self._clip(d_max, delta_g_min, delta_g_max)

        # If DMM exists, try to realize the spread as negative local contribution
        # delta_dmm(T) <= 0 with magnitude <= delta_dmm_max
        if self.dmm and spread > 1e-15 and delta_dmm_max > 0:
            spread = min(spread, delta_dmm_max)
            delta_dmm_T = -spread  # negative
            # weights in [0,1] so that delta_g_T + delta_dmm_T*w_i approximates d_i
            # if we had to clamp spread, this becomes approximate but consistent
            denom = (d_max - d_min) if (d_max - d_min) > 1e-15 else 1.0
            weights = ((d_max - d_i) / denom).clip(0.0, 1.0).tolist()
        else:
            delta_dmm_T = 0.0
            weights = [0.0 for _ in range(Q.shape[0])]

        # Energy scale proxy (in energy units)
        escale = float(np.max(np.abs(d_i))) if d_i.size else 1.0
        if escale <= 1e-15:
            escale = 1.0

        # Choose Omega_max (energy units), then convert to waveform units (/TIME)
        kappa = float(getattr(self.config.drive_shaping, "heuristic_kappa", 0.25))
        omega_max_energy = kappa * escale

        # Convert to waveform amplitude units
        omega_max_wave = omega_max_energy / TIME

        # Respect hardware amplitude constraints
        omega_max_wave = self._scale_omega_for_device_constraints(omega_max_wave)

        # Build waveforms (waveform units)
        eps = 1e-9 / TIME

        # Initial strong negative detuning (easy init)
        delta_0 = delta_g_min
        delta_0 = self._clip(delta_0, delta_g_min, delta_g_max)

        # Convert detunings to waveform units (/TIME)
        delta_0_w = delta_0 / TIME
        delta_g_T_w = delta_g_T / TIME
        delta_dmm_T_w = delta_dmm_T / TIME  # negative or 0

        # Simple 3-phase / 4-point schedule:
        # Amp: 0 -> plateau -> 0
        amp_wave = InterpolatedWaveform(
            max_seq_duration,
            [eps, omega_max_wave, omega_max_wave, eps],
        )

        # Global detuning: stay negative, sweep to final, hold
        det_wave = InterpolatedWaveform(
            max_seq_duration,
            [delta_0_w, delta_0_w, delta_g_T_w, delta_g_T_w],
        )

        # DMM detuning map: off then ramp to final negative shift, then hold
        wdetunings = None
        if self.dmm and delta_dmm_T_w < 0 and any(w > 0 for w in weights):
            wdetunings = weighted_detunings(
                register,
                max_seq_duration,
                weights,                  # IMPORTANT: weights in [0,1]
                final_detuning=delta_dmm_T_w,  # negative
            )

        shaped_drive = Drive(amplitude=amp_wave, detuning=det_wave, weighted_detunings=wdetunings)
        solution = QUBOSolution(torch.Tensor(), torch.Tensor())
        return shaped_drive, solution

class OptimizedDriveShaper(BaseDriveShaper):
    """
    Drive shaper that uses optimization to find the best drive parameters for solving QUBOs.
    Returns an optimized drive, the bitstrings, their counts, probabilities, and costs.

    Attributes:
        drive (Drive): current drive.
        best_cost (float): Current best cost.
        best_bitstring (Tensor | list): Current best bitstring.
        bitstrings (Tensor | list): List of current bitstrings obtained.
        counts (Tensor | list): Frequencies of bitstrings.
        probabilities (Tensor | list): Probabilities of bitstrings.
        costs (Tensor | list): Qubo cost.
        optimized_custom_qubo_cost (Callable[[str, torch.Tensor], float], optional):
            Apply a different qubo cost evaluation during optimization.
            Must be defined as:
            `def optimized_custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`.
            Defaults to None, meaning we use the default QUBO evaluation.
        optimized_custom_objective_fn (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `self.run_simulation` to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = optimized_custom_objective_fn(bitstrings,
                counts, probabilities, costs, best_cost, best_bitstring)`
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        optimized_callback_objective (Callable[..., None], optional): Apply a callback
            during bayesian optimization. Only accepts one input dictionary
            created during optimization `d = {"x": x, "cost_eval": cost_eval}`
            hence should be defined as:
            `def callback_fn(d: dict) -> None:`
            Defaults to None, which means no callback is applied.
    """

    def __init__(
        self,
        instance: QUBOInstance,
        config: SolverConfig,
        backend: concepts.Backend,
    ):
        """Instantiate an `OptimizedDriveShaper`.

        Args:
            instance (QUBOInstance): Qubo instance.
            config (SolverConfig): Configuration for solving.
            backend (Backend): Backend to use during optimization.

        """
        super().__init__(instance, config, backend)

        self.drive = None
        self.best_cost = None
        self.best_bitstring = None
        self.best_params = None
        self.bitstrings = None
        self.counts = None
        self.probabilities = None
        self.costs = None
        self.optimized_custom_qubo_cost = self.config.drive_shaping.optimized_custom_qubo_cost
        self.optimized_custom_objective_fn = self.config.drive_shaping.optimized_custom_objective
        self.optimized_callback_objective = self.config.drive_shaping.optimized_callback_objective

    def generate(
        self,
        register: Register,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate a drive via optimization.

        Args:
            register (Register): The physical register layout.

        Returns:
            Drive: A generated Drive.
            QUBOSolution: An instance of the qubo solution
        """
        # TODO: Harmonize the output of the pulse_shaper generate
        QUBO = self.qubo_coefficients
        self.register = register

        self.norm_weights_list = self._compute_norm_weights()

        n_amp = 3
        n_det = 3
        max_amp: float = 1e6  # large value for bounds if no max_amp
        if self.device._device.channels["rydberg_global"].max_amp:
            max_amp = self.device._device.channels["rydberg_global"].max_amp
            assert max_amp is not None
            # added to avoid rouding errors that make the simulation fail (overcoming max_amp)
            max_amp = max_amp - 1e-6

        max_det: float = 1e6  # large value for bounds if no max_det
        if self.device._device.channels["rydberg_global"].max_abs_detuning:
            max_det = self.device._device.channels["rydberg_global"].max_abs_detuning
            assert max_det is not None
            max_det -= 1e-6  # same

        bounds = [(1, max_amp)] * n_amp + [(-max_det, 0)] + [(-max_det, max_det)] * (n_det - 1)
        x0 = (
            self.config.drive_shaping.optimized_initial_omega_parameters
            + self.config.drive_shaping.optimized_initial_detuning_parameters
        )

        def objective(x: list[float]) -> float:
            drive = self.build_drive(x)

            try:
                bitstrings, counts, probabilities, costs, cost_eval, best_bitstring = (
                    self.run_simulation(
                        self.register,
                        drive,
                        QUBO,
                        convert_to_tensor=False,
                    )
                )
                if self.optimized_custom_objective_fn is not None:
                    cost_eval = self.optimized_custom_objective_fn(
                        bitstrings,
                        counts,
                        probabilities,
                        costs,
                        cost_eval,
                        best_bitstring,
                    )
                if not np.isfinite(cost_eval):
                    print(f"[Warning] Non-finite cost encountered: {cost_eval} at x={x}")
                    cost_eval = 1e4

            except Exception as e:
                print(f"[Exception] Error during simulation at x={x}: {e}")
                cost_eval = 1e4

            if self.optimized_callback_objective is not None:
                self.optimized_callback_objective({"x": x, "cost_eval": cost_eval})
            return float(cost_eval)

        opt_result = gp_minimize(
            objective, bounds, x0=x0, n_calls=self.config.drive_shaping.optimized_n_calls
        )

        if opt_result and opt_result.x:
            self.best_params = opt_result.x
            self.drive = self.build_drive(self.best_params)  # type: ignore[arg-type]

            (
                self.bitstrings,
                self.counts,
                self.probabilities,
                self.costs,
                self.best_cost,
                self.best_bitstring,
            ) = self.run_simulation(self.register, self.drive, QUBO, convert_to_tensor=True)

        if self.bitstrings is None or self.counts is None:
            # TODO: what needs to be returned here?
            # the generate function should always return a drive - even if it is not good.
            # we need to return a drive (self.drive) - which is none here.
            # return self.drive, QUBOSolution(None, None)
            raise RuntimeError("No solution found")

        assert self.costs is not None
        solution = QUBOSolution(
            bitstrings=self.bitstrings,
            counts=self.counts,
            probabilities=self.probabilities,
            costs=self.costs,
        )
        assert self.drive is not None
        return self.drive, solution

    def build_drive(self, params: list) -> Drive:
        """Build the drive from a list of parameters for the objective.

        Args:
            params (list): List of parameters.

        Returns:
            Drive: Drive sequence.
        """
        # enforces AnalogDevice max sequence duration since Digital's has no max duration
        max_seq_duration_ = AnalogDevice.max_sequence_duration
        assert max_seq_duration_ is not None

        TIME, _, _ = self.device.converter.factors
        max_seq_duration = max_seq_duration_ / TIME
        amp_params = [1e-9] + list(params[:3]) + [1e-9]
        det_params = [params[3]] + list(params[4:]) + [params[3]]
        amp_params = [p / TIME for p in amp_params]
        det_params = [p / TIME for p in det_params]

        amp_wave = InterpolatedWaveform(max_seq_duration, amp_params)
        det_wave = InterpolatedWaveform(max_seq_duration, det_params)

        wdetunings = None
        final_detuning = det_params[-1]
        if self.dmm and final_detuning > 0:
            wdetunings = weighted_detunings(
                self.register,
                max_seq_duration,
                self.norm_weights_list,
                final_detuning=-final_detuning,
            )

        shaped_drive = Drive(amplitude=amp_wave, detuning=det_wave, weighted_detunings=wdetunings)

        return shaped_drive

    def compute_qubo_cost(self, bitstring: str, QUBO: torch.Tensor) -> float:
        """The qubo cost for a single bitstring to apply during optimization.

        Args:
            bitstring (str): candidate bitstring.
            QUBO (torch.Tensor): qubo coefficients.

        Returns:
            float: respective cost of bitstring.
        """
        if self.optimized_custom_qubo_cost is None:
            return calculate_qubo_cost(bitstring, QUBO)

        return cast(float, self.optimized_custom_qubo_cost(bitstring, QUBO))

    def run_simulation(
        self,
        register: Register,
        drive: Drive,
        QUBO: torch.Tensor,
        convert_to_tensor: bool = True,
    ) -> tuple:
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
            program = QuantumProgram(register=register, drive=drive)
            program.compile_to(device=self.device)
            execution_result = self.backend.run(program)[0]
            bitstring_counts = execution_result.final_bitstrings

            cost_dict = {b: self.compute_qubo_cost(b, QUBO) for b in bitstring_counts.keys()}

            best_bitstring = min(cost_dict, key=cost_dict.get)  # type: ignore[arg-type]
            best_cost = cost_dict[best_bitstring]

            if convert_to_tensor:
                keys = list(bitstring_counts.keys())
                values = list(bitstring_counts.values())

                bitstrings_tensor = torch.tensor(
                    [[int(b) for b in bitstr] for bitstr in keys], dtype=torch.int32
                )
                counts_tensor = torch.tensor(values, dtype=torch.int32)
                probabilities_tensor = counts_tensor.float() / counts_tensor.sum()

                costs_tensor = torch.tensor(
                    [self.compute_qubo_cost(b, QUBO) for b in keys], dtype=torch.float32
                )

                return (
                    bitstrings_tensor,
                    counts_tensor,
                    probabilities_tensor,
                    costs_tensor,
                    best_cost,
                    best_bitstring,
                )
            else:
                counts = list(bitstring_counts.values())
                nsamples = float(sum(counts))
                return (
                    list(bitstring_counts.keys()),
                    counts,
                    [c / nsamples for c in counts],
                    list(cost_dict.values()),
                    best_cost,
                    best_bitstring,
                )

        except Exception as e:
            print(f"Simulation failed: {e}")
            return (
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                float("inf"),
                None,
            )


def get_drive_shaper(
    instance: QUBOInstance,
    config: SolverConfig,
    backend: concepts.Backend,
) -> BaseDriveShaper:
    """
    Method that returns the correct DriveShaper based on configuration.
    The correct drive shaping method can be identified using the config, and an
    object of this driveshaper can be returned using this function.

    Args:
        instance (QUBOInstance): The QUBO problem to embed.
        config (SolverConfig): The solver configuration used.
        backend (Backend): Backend to extract device from or to use
            during drive shaping.

    Returns:
        (BaseDriveShaper): The representative Drive Shaper object.
    """
    if config.drive_shaping.drive_shaping_method == DriveType.ADIABATIC:
        return AdiabaticDriveShaper(instance, config, backend)
    elif config.drive_shaping.drive_shaping_method == DriveType.HEURISTIC:
        return HeuristicDriveShaper(instance, config, backend)
    elif config.drive_shaping.drive_shaping_method == DriveType.OPTIMIZED:
        return OptimizedDriveShaper(instance, config, backend)
    elif issubclass(config.drive_shaping.drive_shaping_method, BaseDriveShaper):
        return cast(
            BaseDriveShaper,
            config.drive_shaping.drive_shaping_method(instance, config, backend),
        )
    else:
        raise NotImplementedError
