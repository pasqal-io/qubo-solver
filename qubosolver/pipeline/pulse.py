from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
from pulser import Pulse as PulserPulse
from pulser.waveforms import InterpolatedWaveform
from qoolqit._solvers import BaseBackend
from qoolqit._solvers.data import QuantumProgram
from skopt import gp_minimize

from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.qubo_types import PulseType
from qubosolver.utils import calculate_qubo_cost

from .targets import Pulse, Register


class BasePulseShaper(ABC):
    """
    Abstract base class for generating pulse schedules based on a QUBO problem.

    This class transforms the structure of a QUBOInstance into a quantum
    pulse sequence that can be applied to a physical register. The register
    is passed at the time of pulse generation, not during initialization.

    Attributes:
        instance (QUBOInstance): The QUBO problem instance.
        config (SolverConfig): The solver configuration.
        pulse (Pulse, optional): A saved current pulse obtained by `generate`.
        backend (BaseBackend): Backend to use.
        device (Device): Device from backend.

    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: BaseBackend):
        """
        Initialize the pulse shaping module with a QUBO instance.

        Args:
            instance (QUBOInstance): The QUBO problem instance.
            config (SolverConfig): The solver configuration.
            backend (BaseBackend): Backend to use.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.pulse: Pulse | None = None
        self.backend = backend
        self.device = backend.device()

    @abstractmethod
    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Pulse, QUBOSolution]:
        """
        Generate a pulse based on the problem and the provided register.

        Args:
            register (Register): The physical register layout.
            instance (QUBOInstance): The QUBO instance.

        Returns:
            Pulse: A generated pulse object wrapping a Pulser pulse.
            QUBOSolution: An instance of the qubo solution
        """
        pass


# ------- Adiabatic Pulse -------


class AdiabaticPulseShaper(BasePulseShaper):
    """
    A Standard Adiabatic Pulse shaper.
    """

    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Pulse, QUBOSolution]:
        """
        Generate an adiabatic pulse based on the QUBO instance and physical register.

        Args:
            register (Register): The physical register layout for the quantum system.
            instance (QUBOInstance): The QUBO instance.

        Returns:
            tuple[Pulse, QUBOSolution | None]:
                - Pulse: A generated pulse object wrapping a Pulser pulse.
                - QUBOSolution: An instance of the qubo solution
                    - str | None: The bitstring (solution) -> Not computed
                    - float | None: The cost (energy value) -> Not computed
                    - float | None: The probabilities for each bitstring -> Not computed
                    - float | None: The counts of each bitstring -> Not computed
        """

        QUBO = instance.coefficients
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list)
        norm_weights_list = [1 - (w / max_node_weight) for w in weights_list]

        T = 4000
        off_diag = QUBO[
            ~torch.eye(QUBO.shape[0], dtype=bool)
        ]  # Selecting off-diagonal terms of the Qubo with a mask
        rydberg_global = self.device.channels["rydberg_global"]
        assert rydberg_global.max_amp is not None  # FIXME: Document why
        Omega = min(
            torch.max(off_diag).item(),
            rydberg_global.max_amp - 1e-9,
        )

        delta_0 = torch.min(torch.diag(QUBO)).item()
        delta_f = -delta_0

        amp_wave = InterpolatedWaveform(T, [1e-9, Omega, 1e-9])
        det_wave = InterpolatedWaveform(T, [delta_0, 0, delta_f])

        pulser_pulse = PulserPulse(amp_wave, det_wave, 0)
        # PulserPulse has some magic that ensures its constructor does not always return
        # an instance of PulserPulse. Let's make sure (and help mypy realize) that we
        # are building an instance of PulserPulse.
        assert isinstance(pulser_pulse, PulserPulse)

        shaped_pulse = Pulse(pulse=pulser_pulse)
        shaped_pulse.norm_weights = norm_weights_list
        shaped_pulse.duration = T

        self.pulse = shaped_pulse
        solution = QUBOSolution(torch.Tensor(), torch.Tensor())

        return self.pulse, solution


class OptimizedPulseShaper(BasePulseShaper):
    """
    Pulse shaper that uses optimization to find the best pulse parameters for solving QUBOs.
    Returns an optimized pulse, the bitstrings, their counts, probabilities, and costs.

    Attributes:
        pulse (Pulse): current pulse.
        best_cost (float): Current best cost.
        best_bitstring (Tensor | list): Current best bitstring.
        bitstrings (Tensor | list): List of current bitstrings obtained.
        counts (Tensor | list): Frequencies of bitstrings.
        probabilities (Tensor | list): Probabilities of bitstrings.
        costs (Tensor | list): Qubo cost.
        custom_qubo_cost (Callable[[str, torch.Tensor], float], optional):
            Apply a different qubo cost evaluation during optimization.
            Must be defined as:
            `def custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`.
            Defaults to None, meaning we use the default QUBO evaluation.
        custom_objective_fn (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `self.run_simulation` to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = custom_objective_fn(bitstrings,
                counts, probabilities, costs, best_cost, best_bitstring)`
            Defaults to None, which means we optimize using the best cost
            out of the samples.
        callback_objective (Callable[..., None], optional): Apply a callback
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
        """Instantiate an `OptimizedPulseShaper`.

        Args:
            instance (QUBOInstance): Qubo instance.
            config (SolverConfig): Configuration for solving.
            backend (BaseBackend): Backend to use during optimization.

        """
        super().__init__(instance, config, backend)

        self.pulse = None
        self.best_cost = None
        self.best_bitstring = None
        self.best_params = None
        self.bitstrings = None
        self.counts = None
        self.probabilities = None
        self.costs = None
        self.custom_qubo_cost = self.config.pulse_shaping.custom_qubo_cost
        self.custom_objective_fn = self.config.pulse_shaping.custom_objective
        self.callback_objective = self.config.pulse_shaping.callback_objective

    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Pulse, QUBOSolution]:
        """
        Generate a pulse via optimization.

        Args:
            register (Register): The physical register layout.
            instance (QUBOInstance): The QUBO instance.

        Returns:
            Pulse: A generated pulse object wrapping a Pulser pulse.
            QUBOSolution: An instance of the qubo solution
        """
        # TODO: Harmonize the output of the pulse_shaper generate
        QUBO = instance.coefficients
        self.register = register
        self.norm_weights_list = self._compute_norm_weights(QUBO)

        n_amp = 3
        n_det = 3
        max_amp = self.device.channels["rydberg_global"].max_amp
        assert max_amp is not None
        max_amp = max_amp - 1e-6
        # added to avoid rouding errors that make the simulation fail (overcoming max_amp)

        max_det = self.device.channels["rydberg_global"].max_abs_detuning
        assert max_det is not None
        max_det -= 1e-6
        # same

        bounds = [(1, max_amp)] * n_amp + [(-max_det, 0)] + [(-max_det, max_det)] * (n_det - 1)
        x0 = (
            self.config.pulse_shaping.initial_omega_parameters
            + self.config.pulse_shaping.initial_detuning_parameters
        )

        def objective(x: list[float]) -> float:
            pulse = self.build_pulse(x)

            try:
                bitstrings, counts, probabilities, costs, cost_eval, best_bitstring = (
                    self.run_simulation(
                        self.register,
                        pulse,
                        QUBO,
                        convert_to_tensor=False,
                    )
                )
                if self.custom_objective_fn is not None:
                    cost_eval = self.custom_objective_fn(
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

            if self.callback_objective is not None:
                self.callback_objective({"x": x, "cost_eval": cost_eval})
            return float(cost_eval)

        opt_result = gp_minimize(objective, bounds, x0=x0, n_calls=self.config.n_calls)

        if opt_result and opt_result.x:
            self.best_params = opt_result.x
            self.pulse = self.build_pulse(self.best_params)  # type: ignore[arg-type]

            (
                self.bitstrings,
                self.counts,
                self.probabilities,
                self.costs,
                self.best_cost,
                self.best_bitstring,
            ) = self.run_simulation(self.register, self.pulse, QUBO, convert_to_tensor=True)

        if self.bitstrings is None or self.counts is None:
            # TODO: what needs to be returned here?
            # the generate function should always return a pulse - even if it is not good.
            # we need to return a pulse (self.pulse) - which is none here.
            return self.pulse, QUBOSolution(None, None)  # type: ignore[return-value]

        assert self.costs is not None
        solution = QUBOSolution(
            bitstrings=self.bitstrings,
            counts=self.counts,
            probabilities=self.probabilities,
            costs=self.costs,
        )
        assert self.pulse is not None
        return self.pulse, solution

    def _compute_norm_weights(self, QUBO: torch.Tensor) -> list[float]:
        """Compute normalization weights.

        Args:
            QUBO (torch.Tensor): Qubo coefficients.

        Returns:
            list[float]: normalization weights.
        """
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list) if weights_list else 1.0
        norm_weights_list = [
            1 - (w / max_node_weight) if max_node_weight != 0 else 0.0 for w in weights_list
        ]
        return norm_weights_list

    def build_pulse(self, params: list) -> Pulse:
        """Build the pulse from a list of parameters for the objective.

        Args:
            params (list): List of parameters.

        Returns:
            Pulse: pulse sequence.
        """
        amp = InterpolatedWaveform(5000, [1e-9] + list(params[:3]) + [1e-9])
        det = InterpolatedWaveform(5000, [params[3]] + list(params[4:]) + [params[3]])
        pulser_pulse = PulserPulse(amp, det, 0)
        # PulserPulse has some magic that ensures its constructor does not always return
        # an instance of PulserPulse. Let's make sure (and help mypy realize) that we
        # are building an instance of PulserPulse.
        assert isinstance(pulser_pulse, PulserPulse)

        pulse = Pulse(
            pulse=pulser_pulse,
            norm_weights=self.norm_weights_list,
            duration=5000,
        )
        # pulse.pulse.norm_weights = self.norm_weights_list
        # pulse.pulse.duration = 5000
        return pulse

    def compute_qubo_cost(self, bitstring: str, QUBO: torch.Tensor) -> float:
        """The qubo cost for a single bitstring to apply during optimization.

        Args:
            bitstring (str): candidate bitstring.
            QUBO (torch.Tensor): qubo coefficients.

        Returns:
            float: respective cost of bitstring.
        """
        if self.custom_qubo_cost is None:
            return calculate_qubo_cost(bitstring, QUBO)

        return cast(float, self.custom_qubo_cost(bitstring, QUBO))

    def run_simulation(
        self,
        register: Register,
        pulse: Pulse,
        QUBO: torch.Tensor,
        convert_to_tensor: bool = True,
    ) -> tuple:
        """Run a quantum program using backend and returns
            a tuple of (bitstrings, counts, probabilities, costs, best cost, best bitstring).

        Args:
            register (Register): register of quantum program.
            pulse (Pulse): pulse sequence to run on backend.
            QUBO (torch.Tensor): Qubo coefficients.
            convert_to_tensor (bool, optional): Convert tuple components to tensors.
                Defaults to True.

        Returns:
            tuple: tuple of (bitstrings, counts, probabilities, costs, best cost, best bitstring)
        """
        try:
            program = QuantumProgram(
                register=register.register, pulse=pulse.pulse, device=self.device
            )
            bitstring_counts = self.backend.run(program).counts

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


def get_pulse_shaper(
    instance: QUBOInstance,
    config: SolverConfig,
    backend: BaseBackend,
) -> BasePulseShaper:
    """
    Method that returns the correct PulseShaper based on configuration.
    The correct pulse shaping method can be identified using the config, and an
    object of this pulseshaper can be returned using this function.

    Args:
        instance (QUBOInstance): The QUBO problem to embed.
        config (SolverConfig): The solver configuration used.
        backend (BaseBackend): Backend to extract device from or to use
            during pulse shaping.

    Returns:
        (BasePulseShaper): The representative Pulse Shaper object.
    """
    if config.pulse_shaping.pulse_shaping_method == PulseType.ADIABATIC:
        return AdiabaticPulseShaper(instance, config, backend)
    elif config.pulse_shaping.pulse_shaping_method == PulseType.OPTIMIZED:
        return OptimizedPulseShaper(instance, config, backend)
    elif issubclass(config.pulse_shaping.pulse_shaping_method, BasePulseShaper):
        return cast(
            BasePulseShaper,
            config.pulse_shaping.pulse_shaping_method(instance, config, backend),
        )
    else:
        raise NotImplementedError
