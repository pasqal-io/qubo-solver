from __future__ import annotations

from dataclasses import dataclass
import torch

from .enums import Algorithm


@dataclass
class Config():
    """A `ClassicalConfig` instance defines the classical part of a `SolverConfig`.

    Attributes:
        classical_solver_type (solvers.ClassicalAlgorithm, optional): Classical solver type. Defaults to
            `"tabu_search"`.
        cplex_maxtime (float, optional): CPLEX maximum runtime in seconds. Defaults to 600s.
        cplex_log_path (str, optional): CPLEX log path. Default to `solver.log`.
        max_iter (int, optional): Maximum number of iterations to perform for simulated annealing or tabu search.
        max_bitstrings (int, optional): Maximal number of bitstrings returned as solutions.
        sa_initial_temp (float, optional): Starting temperature (controls exploration).
        sa_final_temp (float, optional): Minimum temperature threshold for stopping.
        sa_cooling_rate (float, optional): Cooling rate - should be slightly below 1 (e.g., 0.95–0.99).
        sa_seed (int, optional): Random seed for reproducibility.
        sa_start (torch.Tensor | None, optional): Optional initial bitstring of shape (n,).
        sa_time_limit (float): Maximum runtime in seconds for simulated annealing.
            Defaults to `float("inf")`, meaning no time limit.
        tabu_x0 (torch.Tensor | None, optional): The initial binary solution tensor of shape (n,).
        tabu_tenure (int, optional): Number of iterations a move (bit flip) remains tabu.
        tabu_max_no_improve (int, optional): Maximum number of consecutive iterations
            without improvement before termination.
        tabu_time_limit (float): Maximum execution time for tabu search,
            in seconds. Defaults to `float("inf")`, meaning no time limit.
    """

    algorithm: str | ClassicalAlgorithm = "tabu_search"

    cplex_maxtime: float = 600.0
    cplex_log_path: str = ""

    max_iter: int = 100
    max_bitstrings: int = 1

    sa_initial_temp: float = 10.0
    sa_final_temp: float = 0.1
    sa_cooling_rate: float | None = None
    sa_seed: int | None = None
    sa_start: torch.Tensor | None = None
    sa_time_limit: float = float("inf")

    tabu_x0: torch.Tensor | None = None
    tabu_tenure: int = 7
    tabu_max_no_improve: int = 20
    tabu_time_limit: float = float("inf")

    def __post_init__(self) -> None:
        self.algorithm = self._normalize_classical_solver_type(self.algorithm)

    @staticmethod
    def _normalize_classical_solver_type(
        val: str | ClassicalAlgorithm,
    ) -> ClassicalAlgorithm:
        """Normalize the classical_solver_type attribute."""
        if isinstance(val, ClassicalAlgorithm):
            return val
        elif isinstance(val, str):
            try:
                return ClassicalAlgorithm[val.upper()]
            except KeyError:
                raise ValueError(f"Invalid classical algorithm '{val}'.")
        else:
            raise TypeError("Invalid classical algorithm type.")
