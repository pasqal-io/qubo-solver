# %%writefile mysterious_embedder.py
from __future__ import annotations

import typing

from qubosolver.config import EmbeddingConfig, SolverConfig
from qubosolver.pipeline.embedder import BaseEmbedder
from qubosolver.pipeline.targets import Register as TargetRegister
from qubosolver.solver import QuboSolver


class SetPartitioningEmbedder(BaseEmbedder):
    """
    Custom embedder that forwards to the built-in 'freespace' space-embedder.
    - Reads optional params from self.config.embedding:
        * dimensions: list[int], e.g., [5,4,3,2]
        * steps_per_round_space_embedder: int (default 300)
    - Uses the SAME instance and backend device injected by the solver.
    - Returns a TargetRegister(device, register) built from the freespace result.
    """

    @typing.no_type_check
    def embed(self) -> TargetRegister:
        # 1) Read optional params from the current embedding config

        # 2) Build a temporary config that requests 'freespace' embedding
        freespace_settings = EmbeddingConfig(
            embedding_method="freespace",
        )
        tmp_cfg = SolverConfig.from_kwargs(
            use_quantum=True,
            embedding=freespace_settings,
        )

        # 3) Delegate to the native freespace embedder on the SAME instance
        tmp_solver = QuboSolver(self.instance, tmp_cfg)
        geometry = tmp_solver.embedding()  # has a .register

        # 4) Return a TargetRegister as required by the pipeline
        return geometry
