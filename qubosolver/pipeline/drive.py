from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
from skopt import gp_minimize
import math
import warnings

from qoolqit import Register, Drive, Device
from qoolqit.waveforms import Interpolated as InterpolatedWaveform
from qubosolver import concepts


from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.program import create_compiled_program
from qubosolver.qubo_types import DriveType
from qubosolver.utils import calculate_qubo_cost
from qubosolver.pipeline.waveforms import constant_weighted_dmm


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
            (1 - (w / max_node_weight)) if max_node_weight != 0 else 0.0 for w in weights_list
        ]
        return norm_weights_list

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


def _compare_specs(a: dict, b: dict) -> None:
    """
    Compare two device-spec dictionaries and emit warnings on any discrepancy.

    Checks that both dicts share the same keys and that every common numeric
    value satisfies ``math.isclose``. Emits a ``UserWarning`` for:
    - Keys present in ``a`` but missing in ``b`` (and vice-versa).
    - Keys whose values differ by more than floating-point tolerance.
    ``None`` values on either side are silently skipped.

    Args:
        a (dict): First spec dictionary (e.g. from ``_pulser_specs``).
        b (dict): Second spec dictionary (e.g. from ``_qoolqit_specs``).

    Returns:
        None
    """
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    if keys_a != keys_b:
        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        if only_in_a:
            warnings.warn(f"Keys present in a but missing in b: {only_in_a}")
        if only_in_b:
            warnings.warn(f"Keys present in b but missing in a: {only_in_b}")

    for key in keys_a & keys_b:
        v1 = a[key]
        v2 = b[key]
        if v1 is None or v2 is None:
            continue
        if not math.isclose(float(v1), float(v2)):
            warnings.warn(f"Value mismatch for key '{key}': a={v1}, b={v2}")


def _pulser_specs(
    device: Device, normalize: bool = False, check_against_qoolqit: bool = False
) -> dict[str, float | None]:
    """
    Extract device specs directly from the underlying Pulser device object.

    Reads the ``rydberg_global`` channel and optional DMM channels to build a
    flat dictionary of hardware limits. Optionally normalises all values to
    dimensionless units using the minimum inter-atom distance ``r0`` and the
    corresponding interaction strength ``J0 = C6 / r0^6``, and optionally
    cross-checks the result against ``_qoolqit_specs``.

    Normalisation conventions (when ``normalize=True``):
        - Distances divided by ``r0``.
        - Energies/frequencies divided by ``J0``.
        - Durations divided by ``1000 / J0`` (converting rad/µs → ns).

    Args:
        device (Device): Qoolqit device wrapper whose ``._device`` attribute
            is a Pulser ``Device`` object.
        normalize (bool): If True, return values in dimensionless units.
            Defaults to False (SI units as reported by Pulser).
        check_against_qoolqit (bool): If True, call ``_compare_specs`` to
            compare the result against ``_qoolqit_specs`` and warn on any
            mismatch. Defaults to False.

    Returns:
        dict[str, float | None]: Dictionary with keys:
            - ``max_duration``: Maximum pulse sequence duration.
            - ``max_amplitude``: Maximum Rabi frequency.
            - ``max_abs_detuning``: Maximum absolute detuning.
            - ``min_distance``: Minimum inter-atom distance.
            - ``max_radial_distance``: Maximum radial trap distance.
            - ``min_avg_amp``: Minimum average amplitude.
            - ``dmm_bottom_detuning``: Bottom detuning limit of the first DMM
              channel, or ``None`` if no DMM channels are present.
    """
    pulser_device = device._device
    channel = pulser_device.channels["rydberg_global"]
    specs: dict[str, float | None] = {}
    specs["max_duration"] = pulser_device.max_sequence_duration
    specs["max_amplitude"] = channel.max_amp
    specs["max_abs_detuning"] = channel.max_abs_detuning
    specs["min_distance"] = pulser_device.min_atom_distance
    specs["max_radial_distance"] = pulser_device.max_radial_distance
    specs["min_avg_amp"] = channel.min_avg_amp
    specs["dmm_bottom_detuning"] = None

    dmm_channels = list(getattr(pulser_device, "dmm_channels", {}).values())
    if dmm_channels:
        specs["dmm_bottom_detuning"] = getattr(dmm_channels[0], "bottom_detuning", None)

    if not normalize:
        return specs

    def _normalize(name: str, scale: float) -> None:
        if specs[name] is not None:
            specs[name] /= scale  # type: ignore[operator]

    r0 = specs["min_distance"]
    assert r0 is not None
    _normalize("min_distance", r0)
    _normalize("max_radial_distance", r0)

    C6 = pulser_device.interaction_coeff
    J0 = C6 / (r0**6)

    _normalize("max_amplitude", J0)
    _normalize("max_abs_detuning", J0)
    _normalize("min_avg_amp", J0)
    _normalize("dmm_bottom_detuning", J0)

    # J0 is in rad/us and t in ns, hence the factor 1000
    _normalize("max_duration", 1000.0 / J0)

    if check_against_qoolqit:
        qq_specs = _qoolqit_specs(device)
        _compare_specs(specs, qq_specs)

    return specs


def _qoolqit_specs(
    device: Device, complete_with_pulser: bool = False, check_against_pulser: bool = False
) -> dict[str, float | None]:
    """
    Retrieve normalised device specs from the Qoolqit device abstraction.

    Starts from ``device.specs`` (which are already in dimensionless Qoolqit
    units) and fills in any keys that Qoolqit does not expose natively
    (``min_avg_amp``, ``dmm_bottom_detuning``) either from the Pulser layer
    or with ``None``.

    Args:
        device (Device): Qoolqit device wrapper.
        complete_with_pulser (bool): If True, missing keys are taken from
            ``_pulser_specs(device, normalize=True)`` instead of being set to
            ``None``. Defaults to False.
        check_against_pulser (bool): If True, call ``_compare_specs`` to
            compare the result against the normalised Pulser specs and warn
            on any mismatch. Defaults to False.

    Returns:
        dict[str, float | None]: Augmented spec dictionary in the same
            dimensionless units as ``device.specs``, with at least the keys:
            - ``min_avg_amp``
            - ``dmm_bottom_detuning``
            plus all keys already present in ``device.specs``.
    """
    specs = device.specs
    pulser_specs = _pulser_specs(device, normalize=True)

    def import_from_pulser_or_set(name: str, fallback: float | None = None) -> None:
        if name in specs.keys():
            return
        if complete_with_pulser:
            specs[name] = pulser_specs.get(name, fallback)
        else:
            specs[name] = fallback

    # Don't merge, update specific keys that are known not be in Qoolqit specs
    import_from_pulser_or_set("min_avg_amp")
    import_from_pulser_or_set("dmm_bottom_detuning")

    if check_against_pulser:
        _compare_specs(specs, pulser_specs)

    return specs


class HeuristicDriveShaper(BaseDriveShaper):
    """
    Heuristic schedule drive shaper.

    With DMM:
        Final target encoding:
            d_i = -0.5 * Q_ii

        DMM convention in this stack:
            WeightedDetuning waveform must be <= 0

        Hence we encode the local final detuning as:
            delta_i(T) = delta_g(T) + delta_dmm(T) * w_i

        with:
            delta_g(T)   = d_max
            delta_dmm(T) = -(d_max - d_min) <= 0
            w_i          = (d_max - d_i) / (d_max - d_min) in [0, 1]

        so that:
            delta_i(T) = d_i

    Without DMM:
        Only a global detuning is available, so the final detuning is chosen as:
            delta_g(T) = mean(d_i)
        and no weighted detunings are declared.
    """

    def generate(self, register: Register) -> tuple[Drive, QUBOSolution]:

        # Heuristic coefficient for omega
        kappa = self.config.drive_shaping.heuristic_kappa
        # Hardware bounds
        specs = self.device.specs
        max_seq_duration: float = specs["max_duration"] or 1000.0
        pulser_specs = _pulser_specs(self.device)
        use_dmm = self.dmm and (pulser_specs["dmm_bottom_detuning"] is not None)

        if specs.get("max_amplitude") is not None and specs.get("max_abs_detuning") is not None:
            max_amplitude = specs["max_amplitude"]
            max_abs_detuning = specs["max_abs_detuning"]

            det_amp_ratio = max_amplitude / max_abs_detuning  # type: ignore
            if kappa < det_amp_ratio:
                warnings.warn(
                    f"heuristic_kappa is too small ({kappa}), you're likely to get a qoolqit CompilationError. Set it above {det_amp_ratio}."
                )

        Q = self.qubo_coefficients
        n = self.instance.size

        # Target local final detunings
        d = (-0.5 * torch.diag(Q)).cpu().numpy()
        d_min = np.min(d)
        d_max = np.max(d)

        if use_dmm:
            # Final global detuning is the top value, DMM pulls down locally
            delta_g_T = d_max

            # DMM convention required by WeightedDetuning:
            # waveform must be <= 0
            spread = max(0.0, d_max - d_min)
            # if spread > 1e-15 and delta_dmm_max > 0.0:
            if spread > 1e-15:
                delta_dmm_T = -spread  # must be <= 0
                denom = d_max - d_min
                weights = ((d_max - d) / denom).clip(0.0, 1.0).tolist()
            else:
                use_dmm = False
                delta_dmm_T = 0.0
                weights = [0.0] * n
        else:
            # No DMM: use a single global final detuning
            delta_g_T = np.mean(d)
            delta_dmm_T = 0.0
            weights = [0.0] * n

        omega_max = kappa * np.max(np.abs(d))
        # How to get max detuning ?
        delta_0 = -np.max(np.abs(d))

        # Amplitude waveform: 0 -> plateau -> 0
        eps = 1e-9
        amp_wave = InterpolatedWaveform(
            max_seq_duration,
            [eps, omega_max, omega_max, eps],
        )

        # Global detuning waveform: initial negative -> final target
        det_wave = InterpolatedWaveform(
            max_seq_duration,
            [delta_0, delta_0, delta_g_T, delta_g_T],
        )

        # DMM weighted detunings
        dmm = None
        if use_dmm:
            energy_scale = self.device._target_amp / omega_max
            dmm = constant_weighted_dmm(
                register,
                max_seq_duration,
                weights,
                final_detuning=delta_dmm_T,
                device=self.device,
                energy_scale=energy_scale,
            )

        shaped_drive = Drive(
            amplitude=amp_wave,
            detuning=det_wave,
            dmm=dmm,
        )
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
        Generate a drive via Bayesian optimisation.

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

        eps = 0.0001
        zero = eps
        one = 1.0 - eps

        bounds = (
            [(zero, one)] * n_amp + [(-one, -zero)] + [(-one, one)] * (n_det - 2) + [(zero, one)]
        )
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
            objective,
            bounds,
            x0=x0,
            n_calls=self.config.drive_shaping.optimized_n_calls,
            random_state=self.config.drive_shaping.optimized_seed,
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
        """Build the drive waveform from a normalised parameter vector.

        Args:
            params (list): 6 values — 3 amplitude breakpoints then 3 detuning
                breakpoints, all normalised to [0, 1] (or [-1, 1] for detuning).

        Returns:
            Drive: Drive sequence.
        """
        specs = self.device.specs
        max_seq_duration: float = specs["max_duration"] or 1e3
        max_amplitude: float = specs["max_amplitude"] or 1e4
        max_detuning: float = specs["max_abs_detuning"] or 1e4

        amp_params = [1e-9] + list(params[:3]) + [1e-9]
        det_params = list(params[3:])
        amp_params = [p * max_amplitude for p in amp_params]
        det_params = [p * max_detuning for p in det_params]

        amp_wave = InterpolatedWaveform(max_seq_duration, amp_params)
        det_wave = InterpolatedWaveform(max_seq_duration, det_params)

        dmm = None
        final_detuning = det_params[-1]
        if self.dmm and final_detuning > 0:
            amp_max = amp_wave.max()
            energy_scale = self.device._target_amp / amp_max
            dmm = constant_weighted_dmm(
                self.register,
                max_seq_duration,
                self.norm_weights_list,
                final_detuning=-final_detuning,
                device=self.device,
                energy_scale=energy_scale,
            )

        shaped_drive = Drive(amplitude=amp_wave, detuning=det_wave, dmm=dmm)

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
            program = create_compiled_program(
                device=self.device, config=self.config, drive=drive, embedding=register
            )
            job = self.backend.run(program)
            bitstring_counts = job.results().final_bitstrings

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
    if config.drive_shaping.drive_shaping_method == DriveType.HEURISTIC:
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
