from __future__ import annotations

import typing
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings

from qoolqit import Register
from qoolqit.devices import Device
from qoolqit.embedding.matrix_embedder import Blade, BladeConfig


from qubosolver import QUBOInstance, concepts
from qubosolver.algorithms.greedy.greedy import Greedy
from qubosolver.config import EmbedderType, SolverConfig
from qubosolver.utils.density import calculate_density

warnings.filterwarnings("ignore", module="pulser")


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the QUBO instance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig, backend: concepts.Backend):
        """
        Args:
            instance (QUBOInstance): The QUBO problem to embed.
            config (SolverConfig): The Solver Configuration.
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
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        ...


class BLaDEmbedder(BaseEmbedder):

    def embed(self) -> Register:

        embed_config = self.config.embedding
        default = BladeConfig()
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

        config = BladeConfig(
            steps_per_round=step_per_round,
            starting_positions=starting_positions,
            dimensions=tuple(embed_config.blade_dimensions),
            max_min_dist_ratio=max_min_dist_ratio,
            device=device,
        )

        _blade = Blade(config)
        graph = _blade.embed(self.instance.coefficients.numpy())
        if min_distance is not None:
            graph.rescale_coords(spacing=min_distance)
        register = Register.from_graph(graph)

        return register


class GreedyEmbedder(BaseEmbedder):
    """Create an embedding in a greedy fashion.

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. C / ||r_i - r_j||^6).
    """

    @staticmethod
    def _number_of_traps_from_device(device: Device) -> int:
        """Determine the number of traps to use based on the device constraints.

        Inspects the device's layout and atom number limits to derive an
        appropriate trap count. The resolution order is:

        1. ``max_layout_traps`` – if the device exposes a hard trap limit, use it directly.
        2. ``max_atom_num`` / ``max_layout_filling`` – if only an atom-number limit is
           available, derive the minimum number of traps needed to accommodate that
           many atoms at the device's maximum filling ratio.
        3. Fallback – return ``200`` when neither property is set.

        Args:
            device (Device): The quantum device whose constraints are inspected.

        Returns:
            int: The number of traps to allocate for the embedding.
        """

        if device._device.max_layout_traps:
            return device._device.max_layout_traps

        if device._device.max_atom_num:
            return int(np.ceil(device._device.max_atom_num / device._device.max_layout_filling))

        return 200

    @typing.no_type_check
    def embed(self) -> Register:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        if self.config.embedding.greedy_traps == -1:
            self.config.embedding.greedy_traps = self._number_of_traps_from_device(
                self.config.device
            )

        if self.config.embedding.greedy_traps < self.instance.size:
            raise ValueError(
                "Number of traps must be at least equal to the number of atoms on the register."
            )

        # compute density (unchanged)
        self.config.embedding.greedy_density = calculate_density(
            self.instance.coefficients, self.instance.size
        )

        # build params for the Greedy algorithm
        params = {
            "device": self.config.device._device,
            "layout": self.config.embedding.greedy_layout,
            "traps": int(self.config.embedding.greedy_traps),
            "spacing": float(self.config.embedding.greedy_spacing),
            # animation controls (all read by Greedy)
            "draw_steps": bool(self.config.embedding.draw_steps),  # collect per-step data
            "animation": bool(self.config.embedding.draw_steps),  # render animation after run
            "animation_save_path": self.config.embedding.animation_save_path,  # optional export
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
        min_distance = self.config.embedding.min_distance
        if min_distance is not None:
            min_reg_distance = torch.cdist(coords, coords).fill_diagonal_(float("inf")).min()
            coords *= min_distance / min_reg_distance
        else:
            coords /= self._distance_conversion

        # build the register (unchanged)
        qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
        register = Register(qubits)
        return register


def get_embedder(
    instance: QUBOInstance, config: SolverConfig, backend: concepts.Backend
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
