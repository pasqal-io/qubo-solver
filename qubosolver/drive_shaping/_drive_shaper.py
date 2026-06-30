from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import torch

import qoolqit


from . import heuristic, optimized
from qubosolver.types import QUBOInstance, QUBOSolution, DriveType, _protocols
from qubosolver.config import SolverConfig


class _BaseDriveShaper(ABC):
    """
    Abstract base class for generating Qoolqit drives based on a QUBO problem.

    This class transforms the structure of a QUBOInstance into a quantum
    waveform sequence or drive that can be applied to a physical register. The register
    is passed at the time of drive generation, not during initialization.

    Attributes:
        instance (QUBOInstance): The QUBO problem instance.
        config (SolverConfig): The solver configuration.
        drive (qoolqit.Drive, optional): A saved current drive obtained by `generate`.
        backend (Backend): Backend to use.
        device (Device): Device from backend.

    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: _protocols.Backend):
        """
        Initialize the drive shaping module with a QUBO instance.

        Args:
            instance (QUBOInstance): The QUBO problem instance.
            config (SolverConfig): The solver configuration.
            backend (Backend): Backend to use.
        """
        self.instance: QUBOInstance = instance
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
    ) -> tuple[qoolqit.Drive, QUBOSolution]:
        """
        Generate a drive based on the problem and the provided register.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            qoolqit.Drive: A generated qoolqit.Drive.
            QUBOSolution: An instance of the qubo solution
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

    def generate(self, register: qoolqit.Register) -> tuple[qoolqit.Drive, QUBOSolution]:
        device = self.config.device
        dmm = self.config.drive_shaping.dmm
        # Heuristic coefficient for omega
        kappa = self.config.drive_shaping.heuristic_kappa
        return (
            heuristic.build_drive(
                self.instance, device, dmm=dmm, kappa=kappa, labelling=register.qubits_ids
            ),
            QUBOSolution(),
        )


class OptimizedDriveShaper(_BaseDriveShaper):
    """
    qoolqit.Drive shaper that uses optimization to find the best drive parameters for solving QUBOs.
    Returns an optimized drive, the bitstrings, their counts, probabilities, and costs.

    Attributes:
        drive (qoolqit.Drive): current drive.
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
        optimized_custom_objective (Callable[[list, list, list, list, float, str], float], optional):
            For bayesian optimization, one can change the output of
            `self.run_simulation` to optimize differently. Instead of using the best cost
            out of the samples, one can change the objective for an average,
            or any function out of the form
            `cost_eval = optimized_custom_objective(bitstrings,
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
        backend: _protocols.Backend,
    ):
        """Instantiate an `OptimizedDriveShaper`.

        Args:
            instance (QUBOInstance): Qubo instance.
            config (SolverConfig): Configuration for solving.
            backend (Backend): Backend to use during optimization.

        """
        super().__init__(instance, config, backend)

    def generate(
        self,
        register: qoolqit.Register,
    ) -> tuple[qoolqit.Drive, QUBOSolution]:
        """
        Generate a drive via optimization.

        Args:
            register (qoolqit.Register): The physical register layout.

        Returns:
            qoolqit.Drive: A generated qoolqit.Drive.
            QUBOSolution: An instance of the qubo solution
        """

        config = optimized.Config.from_drive_shaping_config(self.config.drive_shaping)

        return optimized.build_drive(
            self.instance,
            register,
            self.backend,
            self.device,
            dmm=self.dmm,
            config=config,
            labelling=register.qubits_ids,
        )


def _get_drive_shaper(
    instance: QUBOInstance,
    config: SolverConfig,
    backend: _protocols.Backend,
) -> _BaseDriveShaper:
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
        (BaseDriveShaper): The representative qoolqit.Drive Shaper object.
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
