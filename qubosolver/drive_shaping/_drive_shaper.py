from __future__ import annotations

from abc import ABC, abstractmethod

import torch

import qoolqit


from . import (
    proportional_diagonal,
    local_energy_scale,
    bayesian_search,
)
from .enums import Algorithm
from qubosolver.types import Instance, Solution, protocols
from qubosolver import solvers


class _BaseDriveShaper(ABC):
    """Abstract base class for generating Qoolqit drives based on a QUBO problem.

    This class transforms the structure of an `Instance` into a quantum
    waveform sequence or drive that can be applied to a physical register. The
    register is passed at the time of drive generation, not during
    initialization.

    Attributes:
        instance (Instance): The QUBO problem instance.
        config (solvers.QuantumConfig): The solver configuration.
        backend (Backend): Backend to use.
        device (Device): Device from backend.
    """

    def __init__(self, instance: Instance, config: solvers.QuantumConfig, backend: protocols.Backend):
        """Initialize the drive shaping module with a QUBO instance.

        Args:
            instance (Instance): The QUBO problem instance.
            config (solvers.QuantumConfig): The solver configuration.
            backend (Backend): Backend to use.
        """
        self.instance: Instance = instance
        self.config: solvers.QuantumConfig = config
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


class ProportionalDiagonalDriveShaper(_BaseDriveShaper):
    r"""Proportional-diagonal schedule drive shaper.

    With DMM, the final target encoding is $d_i = -\alpha Q_{ii}$. Since the
    DMM convention in this stack requires the weighted-detuning waveform to
    stay $\le 0$, the local final detuning is encoded as

    $$\delta_i(T) = \delta_g(T) + \delta_{\text{dmm}}(T) \cdot w_i$$

    with

    $$\delta_g(T) = d_{\max}, \quad
    \delta_{\text{dmm}}(T) = -(d_{\max} - d_{\min}) \le 0, \quad
    w_i = \frac{d_{\max} - d_i}{d_{\max} - d_{\min}} \in [0, 1]$$

    so that $\delta_i(T) = d_i$.

    Without DMM, only a global detuning is available, so the final detuning
    is chosen as $\delta_g(T) = \text{mean}(d_i)$ and no weighted detunings
    are declared.
    """

    def generate(self, register: qoolqit.Register) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive using the proportional-diagonal schedule.

        Builds amplitude and detuning waveforms from the QUBO coefficients.
        When DMM is available, per-qubit detuning weights are encoded via a
        weighted detuning channel; otherwise a single global detuning is used.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            tuple[qoolqit.Drive, Solution]: The proportional-diagonal drive
            and an empty QUBO solution (no optimization is performed).
        """
        device = self.config.device
        dmm = self.config.drive_shaping.dmm
        # Proportional-diagonal coefficient for omega
        kappa = self.config.drive_shaping.proportional_diagonal_kappa
        return (
            proportional_diagonal.build_drive(
                self.instance, register, device=device, dmm=dmm, kappa=kappa
            ),
            Solution(),
        )


class LocalEnergyScaleDriveShaper(_BaseDriveShaper):
    r"""Local-energy-scale heuristic drive shaper.

    The peak Rabi frequency is proportional to the average local physical
    energy scale,

    $$E_i = |\delta_i(T)| + \sum_{j \neq i} |V_{ij}|,$$

    according to

    $$\omega_{\max} = \kappa \cdot \text{mean}_i(E_i).$$

    No numerical pulse optimization is performed.
    """

    def generate(
        self,
        register: qoolqit.Register,
    ) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive using the local-energy-scale heuristic.

        Args:
            register: Physical register on which the drive will run.

        Returns:
            The generated drive and an empty solution, since no classical
            pulse-parameter optimization is performed.
        """
        device = self.config.device
        dmm = self.config.drive_shaping.dmm
        kappa = self.config.drive_shaping.local_energy_scale_kappa

        return local_energy_scale.build_drive(self.instance, register, device=device, dmm=dmm, kappa=kappa), Solution()


class BayesianSearchDriveShaper(_BaseDriveShaper):
    """Drive shaper that uses Bayesian search to find the best drive parameters for solving QUBOs.

    Generating a drive runs a Bayesian optimization loop over the QUBO cost
    and returns the optimized drive together with a `Solution` holding the
    sampled bitstrings, their counts, probabilities, and costs.
    """

    def __init__(
        self,
        instance: Instance,
        config: solvers.QuantumConfig,
        backend: protocols.Backend,
    ):
        """Instantiate a `BayesianSearchDriveShaper`.

        Args:
            instance (Instance): Qubo instance.
            config (solvers.Config): Configuration for solving.
            backend (Backend): Backend to use during optimization.

        """
        super().__init__(instance, config, backend)

    def generate(
        self,
        register: qoolqit.Register,
    ) -> tuple[qoolqit.Drive, Solution]:
        """Generate a drive via Bayesian search.

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

        config = bayesian_search.Config.from_drive_shaping_config(self.config.drive_shaping)

        return bayesian_search.build_drive(
            self.instance,
            register,
            backend=self.backend,
            device=self.device,
            dmm=self.dmm,
            config=config,
        )


def _get_drive_shaper(
    instance: Instance,
    config: solvers.QuantumConfig,
    backend: protocols.Backend,
) -> _BaseDriveShaper:
    """Return the appropriate drive shaper for the given configuration.

    Selects and instantiates a :class:`_BaseDriveShaper` subclass based on
    ``config.drive_shaping.algorithm``. Supports the built-in
    :class:`ProportionalDiagonalDriveShaper`,
    :class:`LocalEnergyScaleDriveShaper`, and
    :class:`BayesianSearchDriveShaper`.

    Args:
        instance (Instance): The QUBO problem to solve.
        config (solvers.Config): The solver configuration used.
        backend (Backend): Backend to extract device from or to use
            during drive shaping.

    Returns:
        The instantiated drive shaper.

    Raises:
        NotImplementedError: If the configured method is not a recognized
            :class:`Algorithm`.
    """
    algorithm = config.drive_shaping.algorithm
    match algorithm:
        case Algorithm.PROPORTIONAL_DIAGONAL:
            return ProportionalDiagonalDriveShaper(instance, config, backend)
        case Algorithm.LOCAL_ENERGY_SCALE:
            return LocalEnergyScaleDriveShaper(instance, config, backend)
        case Algorithm.BAYESIAN_SEARCH:
            return BayesianSearchDriveShaper(instance, config, backend)
        case _:
            raise NotImplementedError(f"Unsupported drive shaping method: {algorithm!r}")
