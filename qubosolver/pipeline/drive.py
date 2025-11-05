from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
from skopt import gp_minimize

from pulser.devices import AnalogDevice
from qoolqit import Register, QuantumProgram, Drive
from qoolqit.execution.backend import BaseBackend
from qoolqit.waveforms import Interpolated as InterpolatedWaveform


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
        backend (BaseBackend): Backend to use.
        device (Device): Device from backend.

    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: BaseBackend):
        """
        Initialize the drive shaping module with a QUBO instance.

        Args:
            instance (QUBOInstance): The QUBO problem instance.
            config (SolverConfig): The solver configuration.
            backend (BaseBackend): Backend to use.
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

    @abstractmethod
    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate a drive based on the problem and the provided register.

        Args:
            register (Register): The physical register layout.
            instance (QUBOInstance): The QUBO instance.

        Returns:
            Drive: A generated Drive.
            QUBOSolution: An instance of the qubo solution
        """
        pass


class AdiabaticDriveShaper(BaseDriveShaper):
    """
    A Standard Adiabatic Drive shaper.
    """

    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate an adiabatic drive based on the QUBO instance and physical register.

        Args:
            register (Register): The physical register layout for the quantum system.
            instance (QUBOInstance): The QUBO instance.

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
        TIME, _, _ = self.device.converter.factors

        QUBO = instance.coefficients
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list)
        norm_weights_list = [(1 - (w / max_node_weight)) / TIME for w in weights_list]

        off_diag = QUBO[
            ~torch.eye(QUBO.shape[0], dtype=torch.bool)
        ]  # Selecting off-diagonal terms of the Qubo with a mask

        rydberg_global = self.device._device.channels["rydberg_global"]

        Omega = min(
            torch.max(off_diag).item(),
            rydberg_global.max_amp - 1e-9,
        )

        delta_0 = torch.min(torch.diag(QUBO)).item()
        delta_f = -delta_0

        # enforces AnalogDevice max sequence duration since Digital's has no max duration
        max_seq_duration = AnalogDevice.max_sequence_duration
        assert max_seq_duration is not None

        max_seq_duration /= TIME
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
        backend: BaseBackend,
    ):
        """Instantiate an `OptimizedDriveShaper`.

        Args:
            instance (QUBOInstance): Qubo instance.
            config (SolverConfig): Configuration for solving.
            backend (BaseBackend): Backend to use during optimization.

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
        instance: QUBOInstance,
    ) -> tuple[Drive, QUBOSolution]:
        """
        Generate a drive via optimization.

        Args:
            register (Register): The physical register layout.
            instance (QUBOInstance): The QUBO instance.

        Returns:
            Drive: A generated Drive.
            QUBOSolution: An instance of the qubo solution
        """
        # TODO: Harmonize the output of the pulse_shaper generate
        QUBO = instance.coefficients
        self.register = register

        self.norm_weights_list = self._compute_norm_weights(QUBO)

        n_amp = 3
        n_det = 3
        max_amp = self.device._device.channels["rydberg_global"].max_amp
        assert max_amp is not None
        max_amp = max_amp - 1e-6
        # added to avoid rouding errors that make the simulation fail (overcoming max_amp)

        max_det = self.device._device.channels["rydberg_global"].max_abs_detuning
        assert max_det is not None
        max_det -= 1e-6
        # same

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
            # we need to return a drive (self.srive) - which is none here.
            return self.drive, QUBOSolution(None, None)  # type: ignore[return-value]

        assert self.costs is not None
        solution = QUBOSolution(
            bitstrings=self.bitstrings,
            counts=self.counts,
            probabilities=self.probabilities,
            costs=self.costs,
        )
        assert self.drive is not None
        return self.drive, solution

    def _compute_norm_weights(self, QUBO: torch.Tensor) -> list[float]:
        """Compute normalization weights.

        Args:
            QUBO (torch.Tensor): Qubo coefficients.

        Returns:
            list[float]: normalization weights.
        """
        TIME, _, _ = self.device.converter.factors
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list) if weights_list else 1.0
        norm_weights_list = [
            (1 - (w / max_node_weight)) / TIME if max_node_weight != 0 else 0.0
            for w in weights_list
        ]
        return norm_weights_list

    def build_drive(self, params: list) -> Drive:
        """Build the drive from a list of parameters for the objective.

        Args:
            params (list): List of parameters.

        Returns:
            Drive: Drive sequence.
        """
        # enforces AnalogDevice max sequence duration since Digital's has no max duration
        max_seq_duration = AnalogDevice.max_sequence_duration
        assert max_seq_duration is not None

        TIME, _, _ = self.device.converter.factors
        max_seq_duration /= TIME
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
    backend: BaseBackend,
) -> BaseDriveShaper:
    """
    Method that returns the correct DriveShaper based on configuration.
    The correct drive shaping method can be identified using the config, and an
    object of this driveshaper can be returned using this function.

    Args:
        instance (QUBOInstance): The QUBO problem to embed.
        config (SolverConfig): The solver configuration used.
        backend (BaseBackend): Backend to extract device from or to use
            during drive shaping.

    Returns:
        (BaseDriveShaper): The representative Drive Shaper object.
    """
    if config.drive_shaping.drive_shaping_method == DriveType.ADIABATIC:
        return AdiabaticDriveShaper(instance, config, backend)
    elif config.drive_shaping.drive_shaping_method == DriveType.OPTIMIZED:
        return OptimizedDriveShaper(instance, config, backend)
    elif issubclass(config.drive_shaping.drive_shaping_method, BaseDriveShaper):
        return cast(
            BaseDriveShaper,
            config.drive_shaping.drive_shaping_method(instance, config, backend),
        )
    else:
        raise NotImplementedError
