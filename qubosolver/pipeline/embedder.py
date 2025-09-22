from __future__ import annotations

import typing
import warnings
from abc import ABC, abstractmethod
import numpy as np
import torch

from pulser.register import Register as PulserRegister
from qoolqit._solvers import BaseBackend

from qubosolver import QUBOInstance
from qubosolver.algorithms.blade.blade import em_blade
from qubosolver.algorithms.greedy.greedy import Greedy
from qubosolver.config import EmbedderType, SolverConfig
from qubosolver.utils.density import calculate_density

from .targets import Register as TargetRegister

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pulser")


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the QUBO instance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: BaseBackend):
        """
        Args:
            instance (QUBOInstance): The QUBO problem to embed.
            config (SolverConfig): The Solver Configuration.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.register: TargetRegister | None = None
        self.backend = backend

    @abstractmethod
    def embed(self) -> TargetRegister:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """


class BLaDEmbedder(BaseEmbedder):
    """
    BLaDE (Balanced Latently Dimensional Embedder)

    Computes positions for nodes so that their interactions according to a device
    approach the desired values at best. The result can be used as an embedding.
    Its prior target is on interaction matrices or QUBOs, but it can also be used
    for MIS with limitations if the adjacency matrix is converted into a QUBO.
    The general principle is based on the Fruchterman-Reingold algorithm.
    """

    @typing.no_type_check
    @staticmethod
    def _preprocessing_qubo(Q: torch.Tensor) -> torch.Tensor:
        def has_empty_positions(X: np.ndarray) -> bool:
            return True if sum([torch.sum(~torch.any(X, dim=0))]) else False

        def get_position_indexes(X: np.ndarray) -> int:
            return np.where(~torch.any(X, dim=0))[-1]

        Q = torch.tensor(Q)

        if has_empty_positions(Q):
            Q_ = Q.detach().clone()

            non_zero_diag = Q.diagonal()[Q.diagonal() != 0]
            min_coeff = non_zero_diag.min() if non_zero_diag.numel() else Q[Q != 0].min()

            for empty_position_index in get_position_indexes(Q):
                Q_[empty_position_index][empty_position_index] = min_coeff * 1e-05

            return torch.abs(Q_ / torch.norm(Q_))

        return torch.abs(Q / torch.norm(Q))

    @typing.no_type_check
    def embed(self) -> TargetRegister:

        coords = em_blade(
            qubo=BLaDEmbedder._preprocessing_qubo(self.instance.coefficients.numpy()),
            device=self.backend.device(),
            draw_steps=self.config.embedding.draw_steps,
            dimensions=self.config.embedding.blade_dimensions,
            starting_positions=(
                self.config.embedding.starting_positions.numpy()
                if self.config.embedding.starting_positions is not None
                else None
            ),
            steps_per_round=self.config.embedding.blade_steps_per_round,
        )

        qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
        register = PulserRegister(qubits)
        return TargetRegister(self.backend.device(), register)


class GreedyEmbedder(BaseEmbedder):
    """Create an embedding in a greedy fashion.

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. C / ||r_i - r_j||^6).
    """

    @typing.no_type_check
    def embed(self) -> TargetRegister:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        if self.config.embedding.traps < self.instance.size:
            raise ValueError(
                "Number of traps must be at least equal to the number of atoms on the register."
            )

        # compute density (unchanged)
        self.config.embedding.density = calculate_density(
            self.instance.coefficients, self.instance.size
        )

        # build params for the Greedy algorithm
        params = {
            "device": self.backend.device(),
            "layout": self.config.embedding.layout_greedy_embedder,
            "traps": int(self.config.embedding.traps),
            "spacing": float(self.config.embedding.spacing),
            # animation controls (all read by Greedy)
            "draw_steps": bool(self.config.embedding.draw_steps),  # collect per-step data
            "animation": bool(self.config.embedding.draw_steps),  # render animation after run
            "animation_save_path": self.config.embedding.animation_save_path,  # optional export
            # "animation_top_k": 5,  # (optional) uncomment if you add support for this in Greedy
        }

        # --- DEBUG / INFO: show where Greedy comes from + the params we’ll pass
        dev = params["device"]
        dev_str = (
            getattr(dev, "name", None)
            or getattr(dev, "device_name", None)
            or dev.__class__.__name__
        )
        printable = dict(params)
        printable["device"] = dev_str  # avoid dumping the whole object
        # --- Call Greedy (unchanged public signature)
        best, _, coords, _, _ = Greedy().launch_greedy(
            Q=self.instance.coefficients,
            params=params,
            # no extra kwargs; Greedy reads animation/draw/save_path from params
        )

        # build the register (unchanged)
        qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
        register = PulserRegister(qubits)
        return TargetRegister(self.backend.device(), register)


def get_embedder(
    instance: QUBOInstance, config: SolverConfig, backend: BaseBackend
) -> BaseEmbedder:
    """
    Method that returns the correct embedder based on configuration.
    The correct embedding method can be identified using the config, and an
    object of this embedding can be returned using this function.

    Args:
        instance (QUBOInstance): The QUBO problem to embed.
        config (Device): The quantum device to target.

    Returns:
        (BaseEmbedder): The representative embedder object.
    """

    if config.embedding.embedding_method == EmbedderType.BLADE:
        return BLaDEmbedder(instance, config, backend)
    elif config.embedding.embedding_method == EmbedderType.GREEDY:
        return GreedyEmbedder(instance, config, backend)
    elif issubclass(config.embedding.embedding_method, BaseEmbedder):
        return typing.cast(
            BaseEmbedder, config.embedding.embedding_method(instance, config, backend)
        )
    else:
        raise NotImplementedError
