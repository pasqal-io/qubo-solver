from __future__ import annotations

import inspect
from typing import Literal, Any
from dataclasses import dataclass, field

import torch

from qubosolver.utils._config import _Config
from . import Algorithm, Layout


@dataclass
class Config(_Config):
    """A `EmbeddingConfig` instance defines the embedding
        part of a `SolverConfig`.

    Attributes:
        embedding_method (str | embedding.Algorithm | type[BaseEmbedder], optional): The type of
            embedding method used to place atoms on the register according to the QUBO problem.
            Defaults to `embedding.Algorithm.GREEDY`.
        greedy_layout (embedding.Layout | str, optional): embedding.Layout type for the
            greedy embedder method. Defaults to `embedding.Layout.TRIANGULAR`.
        greedy_traps (int, optional): The number of traps on the register.
            Defaults to ``-1``, i.e. automatically set to match the selected device capacity.
            A too high value will impede computational efficiency.
        greedy_max_possible_term (float | tuple[Literal['factor'], float]):
            If a float, it corresponds to the maximum representable quadratic
            term. If a tuple, the first element should be 'factor', and the
            second element is a multiplier on the QUBO's maximum quadratic term
            to define the maximum representable quadratic term.
            Defaults to ('factor', 1.0). The maximum possible term corresponds
            to the interaction for the closest possible pair in the layout.
            Setting it to a higher value than the actual maximum increases the
            resolution to represent the terms. Setting it to a lower value
            decreases the resolution and allows traps to be set farther to
            potentially represent smaller terms.
        greedy_density (float | None, optional): The estimated density of the QUBO matrix.
            Defaults to None.
        blade_steps_per_round (int | None): See [Qoolqit's documentation](https://pasqal-io.github.io/qoolqit/main/reference/internals/)
        blade_starting_positions (torch.Tensor | None): See [Qoolqit's documentation](https://pasqal-io.github.io/qoolqit/main/reference/internals/)
        blade_dimensions (list[int]): See [Qoolqit's documentation](https://pasqal-io.github.io/qoolqit/main/reference/internals/)
        draw_steps (bool, optional): Show generated graph at each step of the optimization.
            Defaults to `False`.
        animation_save_path (str | None, optional): If provided, path to save animation.
            Defaults to None.
        max_min_dist_ratio (float | Literal["device"], optional): Maximum allowed ratio
            between the largest and the smallest inter-atom distance in the resulting
            register. When ``"device"``, it is derived from the configured device's
            ``max_radial_distance`` / ``min_distance`` specs. Defaults to ``"device"``.
    """

    algorithm: Algorithm | str = Algorithm.GREEDY

    greedy_layout: Layout = Layout.TRIANGULAR
    greedy_traps: int | Literal["device"] = "device"
    greedy_max_possible_term: float | tuple[Literal["factor"], float] = ("factor", 1.0)
    greedy_density: float | None = None
    blade_steps_per_round: int | None = 200
    blade_starting_positions: torch.Tensor | None = None
    blade_dimensions: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    draw_steps: bool = False
    animation_save_path: str | None = None
    max_min_dist_ratio: float | Literal["device"] = "device"

    def __post_init__(self) -> None:
        self.algorithm = self._normalize_algorithm(self.algorithm)
        self.greedy_layout = self._normalize_layout(self.greedy_layout)

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the fields relevant to the active embedder type.

        Always includes ``embedding_method``, ``draw_steps``, and
        ``animation_save_path``. Additionally includes ``greedy_*`` fields
        when ``embedding_method`` is ``GREEDY``, or ``blade_*`` fields when it
        is ``BLADE``.

        Returns:
            dict[str, Any]: Serialized representation of this config.
        """
        serialization: dict = {
            "algorithm": self.algorithm,
            "draw_steps": self.draw_steps,
            "animation_save_path": self.animation_save_path,
        }
        dict_all_fields = self.__dict__
        if self.algorithm == Algorithm.GREEDY:
            serialization.update(
                {
                    k: v
                    for k, v in dict_all_fields.items()
                    if k.startswith(Algorithm.GREEDY.value)
                }
            )
        if self.algorithm == Algorithm.BLADE:
            serialization.update(
                {k: v for k, v in dict_all_fields.items() if k.startswith(Algorithm.BLADE.value)}
            )
        return serialization

    @staticmethod
    def _normalize_algorithm(val: Any) -> Algorithm | Any:
        """Normalize the embedded attribute."""
        if isinstance(val, Algorithm):
            return val
        elif isinstance(val, str):
            try:
                return Algorithm[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid str embedding method '{val}'.")
        elif inspect.isclass(val):
            from qubosolver.embedding._embedder import _BaseEmbedder

            if not issubclass(val, _BaseEmbedder):
                raise TypeError(f"Class must be a subclass of {_BaseEmbedder.__name__}")
            else:
                return val
        else:
            raise TypeError("Invalid embedding method type.")

    @staticmethod
    def _normalize_layout(val: str | Layout) -> Layout:
        """Normalize the layout attribute."""
        if isinstance(val, Layout):
            return val
        u = val.upper()
        if u == Layout.SQUARE.name:
            return Layout.SQUARE
        elif u == Layout.TRIANGULAR.name:
            return Layout.TRIANGULAR
        else:
            raise ValueError(f"Invalid layout '{val}'.")

