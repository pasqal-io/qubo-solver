from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import torch

import qoolqit


from . import heuristic, optimized
from qubosolver.types import Instance, Solution, DriveType, _protocols
from qubosolver.config import SolverConfig


class _BaseDriveShaper(ABC):
    """
    Abstract base class for generating Qoolqit drives based on a QUBO problem.

    This class transforms the structure of a Instance into a quantum
    waveform sequence or drive that can be applied to a physical register. The register
    is passed at the time of drive generation, not during initialization.

    Attributes:
        instance (Instance): The QUBO problem instance.
        config (SolverConfig): The solver configuration.
        backend (Backend): Backend to use.
        device (Device): Device from backend.

    """

    def __init__(self, instance: Instance, config: SolverConfig, backend: _protocols.Backend):
        """
        Initialize the drive shaping module with a QUBO instance.

        Args:
            instance (Instance): The QUBO problem instance.
            config (SolverConfig): The solver configuration.
            backend (Backend): Backend to use.
        """
        self.instance: Instance = instance
        self.config: SolverConfig = config
        self.backend = backend
        self.device = self.config.device

        # check if device allow DMM
        self.dmm = self.config.drive_shaping.dmm and (
            len(list(self.config.device._device.dmm_channels.keys())) > 0
        )

    @property
    def qubo_coefficients(self) -> torch.Tensor:
        """The raw QUBO coefficient matrix."""
        return self.instance.matrix

    @property
    def qubo_normalized_coefficients(self) -> torch.Tensor:
        """The QUBO coefficient matrix normalized by its maximum off-diagonal value."""
        return self.instance._normalized_matrix

    @abstractmethod
    def generate(
        self,
        register: qoolqit.Register,
    ) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive based on the problem and the provided register.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            tuple[qoolqit.Drive, Solution]: The generated drive and the
            associated QUBO solution.
        """
        pass


class HeuristicDriveShaper(_BaseDriveShaper):
    """
    Heuristic schedule drive shaper.

    With DMM:
        Final target encoding:
            d_i = -alpha * Q_ii

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

    def generate(self, register: qoolqit.Register) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive using the heuristic schedule.

        Builds amplitude and detuning waveforms from the QUBO coefficients.
        When DMM is available, per-qubit detuning weights are encoded via a
        weighted detuning channel; otherwise a single global detuning is used.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            tuple[qoolqit.Drive, Solution]: The heuristic drive and an
            empty QUBO solution (no optimization is performed).
        """
        device = self.config.device
        dmm = self.config.drive_shaping.dmm
        # Heuristic coefficient for omega
        kappa = self.config.drive_shaping.heuristic_kappa
        return (
            heuristic.build_drive(self.instance, device, dmm=dmm, kappa=kappa),
            Solution(),
        )


class OptimizedDriveShaper(_BaseDriveShaper):
    """
    qoolqit.Drive shaper that uses optimization to find the best drive parameters for solving QUBOs.
    Returns an optimized drive, the bitstrings, their counts, probabilities, and costs.
    """

    def __init__(
        self,
        instance: Instance,
        config: SolverConfig,
        backend: _protocols.Backend,
    ):
        """Instantiate an `OptimizedDriveShaper`.

        Args:
            instance (Instance): Qubo instance.
            config (SolverConfig): Configuration for solving.
            backend (Backend): Backend to use during optimization.

        """
        super().__init__(instance, config, backend)

    def generate(
        self,
        register: qoolqit.Register,
    ) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive via optimization.

        Builds drive parameters by running a Bayesian optimization loop
        over the QUBO cost. Supports optional DMM channels and custom
        cost/objective/callback overrides defined in the solver config.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            tuple[qoolqit.Drive, Solution]: The optimized drive and the
            associated QUBO solution containing bitstrings, costs, and
            probabilities from the final simulation run.
        """

        config = optimized.Config.from_drive_shaping_config(self.config.drive_shaping)

        return optimized.build_drive(
            self.instance,
            register,
            self.backend,
            self.device,
            dmm=self.dmm,
            config=config,
        )


def _get_drive_shaper(
    instance: Instance,
    config: SolverConfig,
    backend: _protocols.Backend,
) -> _BaseDriveShaper:
    """Return the appropriate drive shaper for the given configuration.

    Selects and instantiates a :class:`_BaseDriveShaper` subclass based on
    ``config.drive_shaping.drive_shaping_method``. Supports the built-in
    :class:`HeuristicDriveShaper` and :class:`OptimizedDriveShaper`, as well
    as any custom subclass of :class:`_BaseDriveShaper` passed directly in
    the config.

    Args:
        instance (Instance): The QUBO problem to solve.
        config (SolverConfig): The solver configuration used.
        backend (Backend): Backend to extract device from or to use
            during drive shaping.

    Returns:
        _BaseDriveShaper: The instantiated drive shaper.

    Raises:
        NotImplementedError: If ``config.drive_shaping.drive_shaping_method``
            is not a recognised :class:`DriveType` and is not a subclass of
            :class:`_BaseDriveShaper`.
    """
    if config.drive_shaping.drive_shaping_method == DriveType.HEURISTIC:
        return HeuristicDriveShaper(instance, config, backend)
    elif config.drive_shaping.drive_shaping_method == DriveType.OPTIMIZED:
        return OptimizedDriveShaper(instance, config, backend)
    elif issubclass(config.drive_shaping.drive_shaping_method, _BaseDriveShaper):
        return cast(
            _BaseDriveShaper,
            config.drive_shaping.drive_shaping_method(instance, config, backend),
        )
    else:
        raise NotImplementedError
