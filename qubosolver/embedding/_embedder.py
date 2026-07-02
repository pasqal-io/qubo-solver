from __future__ import annotations

import typing
from abc import ABC, abstractmethod
import warnings

from qoolqit import Register

from . import blade, greedy
from qubosolver.types import QUBOInstance, EmbedderType, _protocols
from qubosolver.config import SolverConfig

warnings.filterwarnings("ignore", module="pulser")


class _BaseEmbedder(ABC):
    """Abstract base class for all embedders.

    Subclasses translate a :class:`~qubosolver.types.QUBOInstance` into a
    physical :class:`~qoolqit.Register` — a set of atom positions compatible
    with Pasqal/Pulser devices — by mapping the QUBO graph structure onto a
    2-D trap layout.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: _protocols.Backend):
        """
        Args:
            instance: The QUBO problem to embed.
            config: Solver configuration, including embedding parameters
                (via ``config.embedding``) and device constraints
                (via ``config.device``).
            backend: Execution backend, passed through for embedders that
                need backend-specific information during placement.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.register: Register | None = None
        self.backend = backend

        # TODO: remove when bumping to qoolqit v1
        # for converting to qoolqit
        self._distance_conversion = self.config.device.converter.factors[2]

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

        Reads embedding parameters from ``self.config.embedding`` and device
        limits from ``self.config.device``.  When ``min_distance`` is set in
        the embedding config the resulting register is normalised so that the
        closest atom pair is exactly ``min_distance`` apart; otherwise the
        raw device radial-distance bounds drive the layout directly via
        ``max_min_dist_ratio``.

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

        min_distance = self.config.embedding.min_distance
        max_radial_distance = self.config.device.specs["max_radial_distance"]
        if min_distance is None or max_radial_distance is None:
            device = self.config.device
            max_min_dist_ratio = None
        else:
            device = None
            max_min_dist_ratio = max_radial_distance / min_distance

        config = blade.Config(
            steps_per_round=step_per_round,
            starting_positions=starting_positions,
            dimensions=tuple(embed_config.blade_dimensions),
            max_min_dist_ratio=max_min_dist_ratio,
            device=device,
        )
        return blade.embed(self.instance, config=config, normalize=(min_distance is not None))


class GreedyEmbedder(_BaseEmbedder):
    """Create an embedding in a greedy fashion.

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. C / ||r_i - r_j||^6).
    """

    def embed(self) -> Register:
        """Embed the QUBO instance using the greedy algorithm.

        At each step the algorithm selects the logical node and the available
        trap that minimise the incremental mismatch between the QUBO edge
        weights and the physical interaction matrix (∝ C/‖rᵢ − rⱼ‖⁶).
        When ``min_distance`` is set in the embedding config the register is
        normalised so that the closest atom pair is exactly ``min_distance``
        apart.

        Returns:
            The atom register with positions determined by the greedy placer.
        """
        config = greedy.Config.from_embedding_config(self.config.embedding)
        normalize = self.config.embedding.min_distance is not None
        return greedy.embed(self.instance, self.config.device, config=config, normalize=normalize)


def _get_embedder(
    instance: QUBOInstance, config: SolverConfig, backend: _protocols.Backend
) -> _BaseEmbedder:
    """Return the appropriate embedder instance for the given configuration.

    Inspects ``config.embedding.embedding_method`` and constructs the matching
    :class:`_BaseEmbedder` subclass:

    * :class:`BLaDEmbedder` — when the method is :attr:`EmbedderType.BLADE`.
    * :class:`GreedyEmbedder` — when the method is :attr:`EmbedderType.GREEDY`.
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
            recognised :class:`EmbedderType` value and is not a subclass of
            :class:`_BaseEmbedder`.
    """

    if config.embedding.embedding_method == EmbedderType.BLADE:
        return BLaDEmbedder(instance, config, backend)
    elif config.embedding.embedding_method == EmbedderType.GREEDY:
        return GreedyEmbedder(instance, config, backend)
    elif issubclass(config.embedding.embedding_method, _BaseEmbedder):
        return typing.cast(
            _BaseEmbedder, config.embedding.embedding_method(instance, config, backend)
        )
    else:
        raise NotImplementedError
