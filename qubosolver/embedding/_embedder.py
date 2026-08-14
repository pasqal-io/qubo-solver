from __future__ import annotations

import typing
from abc import ABC, abstractmethod
import warnings

import torch
from qoolqit import Register

from qubosolver import embedding
from . import blade, greedy_layout
from qubosolver.types import Instance, protocols
from qubosolver.config import SolverConfig

warnings.filterwarnings("ignore", module="pulser")


class _BaseEmbedder(ABC):
    """Abstract base class for all embedders.

    Subclasses translate a :class:`~qubosolver.types.Instance` into a
    physical :class:`~qoolqit.Register` — a set of atom positions compatible
    with Pasqal/Pulser devices — by mapping the QUBO graph structure onto a
    2-D trap layout.
    """

    def __init__(self, instance: Instance, config: SolverConfig, backend: protocols.Backend):
        """
        Args:
            instance: The QUBO problem to embed.
            config: Solver configuration, including embedding parameters
                (via ``config.embedding``) and device constraints
                (via ``config.device``).
            backend: Execution backend, passed through for embedders that
                need backend-specific information during placement.
        """
        self.instance: Instance = instance
        self.config: SolverConfig = config
        self.register: Register | None = None
        self.backend = backend

    @abstractmethod
    def embed(self) -> Register:
        """Place atoms and return the resulting register.

        Concrete implementations must translate the QUBO graph into a
        :class:`~qoolqit.Register` whose atom positions respect the
        target device's spatial constraints (radial distance, minimum
        atom separation, etc.).

        Returns:
            The atom register ready for use in a quantum program.
        """
        ...


class BLaDEmbedder(_BaseEmbedder):
    """Embedder using the BLaDE (Balanced Layout and Distance Embedding) algorithm.

    BLaDE iteratively adjusts atom positions so that the physical interaction
    strengths (∝ 1/r⁶) match the QUBO edge weights as closely as possible.
    Embedding parameters (steps per round, initial positions, dimension
    sequence) are read from ``config.embedding``.
    """

    def embed(self) -> Register:
        """Embed the QUBO instance using BLaDE.

        Reads embedding parameters from ``self.config.embedding`` and the
        resolved ``self.config.max_min_dist_ratio`` that bounds the layout's
        largest-to-smallest inter-atom distance ratio.

        Returns:
            The atom register with positions determined by BLaDE.
        """
        embed_config = self.config.embedding
        default = blade.Config()
        step_per_round = embed_config.blade_steps_per_round
        if step_per_round is None:
            step_per_round = default.steps_per_round
        if embed_config.blade_starting_positions is not None:
            starting_positions = embed_config.blade_starting_positions.numpy()
        else:
            starting_positions = None

        max_min_dist_ratio: float | None = self.config.max_min_dist_ratio
        if max_min_dist_ratio == torch.inf:
            max_min_dist_ratio = None

        config = blade.Config(
            steps_per_round=step_per_round,
            starting_positions=starting_positions,
            dimensions=tuple(embed_config.blade_dimensions),
            max_min_dist_ratio=max_min_dist_ratio,
        )
        return blade.embed(self.instance, config=config)


class GreedyEmbedder(_BaseEmbedder):
    """Create an embedding in a greedy fashion.

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. 1 / ||r_i - r_j||^6).
    """

    def embed(self) -> Register:
        """Embed the QUBO instance using the greedy algorithm.

        At each step the algorithm selects the logical node and the available
        trap that minimise the incremental mismatch between the QUBO edge
        weights and the physical interaction matrix (∝ 1/‖rᵢ − rⱼ‖⁶). The
        algorithm operates entirely in adimensional units, so its output
        coordinates require no further normalization.

        Returns:
            The atom register with positions determined by the greedy placer.
        """
        config = greedy_layout.Config._from_embedding_config(self.config.embedding)
        return greedy_layout.embed(self.instance, self.config.device, config=config)


def _get_embedder(
    instance: Instance, config: SolverConfig, backend: protocols.Backend
) -> _BaseEmbedder:
    """Return the appropriate embedder instance for the given configuration.

    Inspects ``config.embedding.embedding_method`` and constructs the matching
    :class:`_BaseEmbedder` subclass:

    * :class:`BLaDEmbedder` — when the method is :attr:`embedding.Algorithm.BLADE`.
    * :class:`GreedyEmbedder` — when the method is :attr:`embedding.Algorithm.GREEDY`.
    * A user-supplied subclass of :class:`_BaseEmbedder` — when the method is
      a class (not a string enum value) that is a subclass of
      :class:`_BaseEmbedder`.

    Args:
        instance: The QUBO problem to embed.
        config: Solver configuration carrying ``config.embedding`` (embedding
            parameters) and ``config.device`` (device constraints).
        backend: Execution backend forwarded to the embedder constructor.

    Returns:
        A concrete :class:`_BaseEmbedder` ready to have :meth:`~_BaseEmbedder.embed` called.

    Raises:
        NotImplementedError: If ``config.embedding.embedding_method`` is not a
            recognised :class:`embedding.Algorithm` value and is not a subclass of
            :class:`_BaseEmbedder`.
    """

    if config.embedding.embedding_method == embedding.Algorithm.BLADE:
        return BLaDEmbedder(instance, config, backend)
    elif config.embedding.embedding_method == embedding.Algorithm.GREEDY:
        return GreedyEmbedder(instance, config, backend)
    elif issubclass(config.embedding.embedding_method, _BaseEmbedder):
        return typing.cast(
            _BaseEmbedder, config.embedding.embedding_method(instance, config, backend)
        )
    else:
        raise NotImplementedError
