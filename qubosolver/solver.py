from __future__ import annotations

from typing import Callable
from collections import Counter

import torch
import random

# Import the classical solver factory from our classical_solver module.
from qoolqit import Register, Drive

from qubosolver.qubo_instance import QUBOInstance
from qubosolver.data import QUBOSolution
from qubosolver.classical_solver import get_classical_solver
from qubosolver.config import SolverConfig, DecompositionConfig
from qubosolver.pipeline import (
    BaseSolver,
    Fixtures,
    get_embedder,
    get_drive_shaper,
)


# Modules to be automatically added to the qubosolver namespace
__all__: list[str] = ["QuboSolver"]


class QuboSolver(BaseSolver):
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        super().__init__(instance, config)
        self._solver: BaseSolver

        if config is None:
            self._solver = QuboSolverClassical(instance, self.config)
        else:
            if config.decompose:
                if self.config.use_quantum:
                    solver_factory: type[BaseSolver] = QuboSolverQuantum
                else:
                    solver_factory = QuboSolverClassical
                self._solver = DecomposeQuboSolver(instance, self.config, solver_factory)

            elif config.use_quantum:
                self._solver = QuboSolverQuantum(instance, config)
            else:
                self._solver = QuboSolverClassical(instance, config)

        self.n_fixed_variables_preprocessing = 0

    def embedding(self) -> Register:
        return self._solver.embedding()

    def drive(self, embedding: Register) -> tuple:
        return self._solver.drive(embedding)

    def solve(self) -> QUBOSolution:
        return self._solver.solve()


class QuboSolverQuantum(BaseSolver):
    """
    Quantum solver that orchestrates the solving of a QUBO problem using
    embedding, drive shaping, and quantum execution pipelines.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        """
        Initialize the QuboSolver with the given problem and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig): Solver settings including backend and device.
        """
        super().__init__(instance, config or SolverConfig(use_quantum=True))
        self._check_size_limit()

        self.fixtures = Fixtures(self.instance, self.config)
        self.backend = self.config.backend
        self.embedder = get_embedder(self.instance, self.config, self.backend)
        self.drive_shaper = get_drive_shaper(self.instance, self.config, self.backend)

        self._register: Register | None = None
        self._drive: Drive | None = None

    def _check_size_limit(self) -> None:
        if (self.instance._coefficients is not None) and self.instance.size > 80:  # type: ignore[operator]
            raise ValueError(
                f"QUBO size {self.instance.size}×{self.instance.size}"
                + " exceeds the maximum supported size of 80×80."
            )

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the QUBO variables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        self.embedder.instance = self.instance
        self._register = self.embedder.embed()
        return self._register

    def drive(self, embedding: Register) -> tuple:
        """
        Generate the drive sequence based on the given embedding.

        Args:
            embedding (Register): The embedded register layout.

        Returns:
            tuple:
                A tuple of
                    - Drive: Drive schedule for quantum execution.
                    - QUBOSolution: Initial solution of generated from drive shaper

        """
        drive, qubo_solution = self.drive_shaper.generate(embedding, self.instance)

        self._drive = drive
        return drive, qubo_solution

    def solve(self) -> QUBOSolution:
        """
        Execute the full quantum pipeline: preprocess, embed, drive, execute, postprocess.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
        """
        # 1) try trivial
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial is not None:
                return trivial
        self._check_size_limit()

        # 2) else delegate to quantum or classical solver
        # Delegate the solving task to the appropriate classical solver using the factory
        if self.config.do_preprocessing:
            # Apply preprocessing and change the solved QUBO by the reduced one
            self.fixtures.preprocess()
            if (
                self.fixtures.reduced_qubo.coefficients is not None
                and len(self.fixtures.reduced_qubo.coefficients) > 0
            ):

                self.instance = self.fixtures.reduced_qubo
                self.n_fixed_variables_preprocessing = self.fixtures.n_fixed_variables

        embedding = self.embedding()

        drive, qubo_solution = self.drive(embedding)

        bitstrings, counts, _, _ = (
            qubo_solution.bitstrings,
            qubo_solution.counts,
            qubo_solution.probabilities,
            qubo_solution.costs,
        )
        if (
            len(bitstrings) == 0 and qubo_solution.counts is None
        ) or self.config.drive_shaping.optimized_re_execute_opt_drive:
            bitstrings, counts = self.execute(drive, embedding)

        bitstring_strs = bitstrings
        bitstrings_tensor = torch.tensor(
            [list(map(int, bs)) for bs in bitstring_strs], dtype=torch.float32
        )
        if counts is None:
            counts_tensor = torch.empty((0,), dtype=torch.int32)
        elif isinstance(counts, dict) or isinstance(counts, Counter):
            count_values = [counts.get(bs, 0) for bs in bitstring_strs]
            counts_tensor = torch.tensor(count_values, dtype=torch.int32)
        else:
            counts_tensor = counts

        solution = QUBOSolution(
            bitstrings=bitstrings_tensor,
            counts=counts_tensor,
            costs=torch.Tensor(),
            probabilities=None,
        )

        # Post-process fixations of the preprocessing and restore the original QUBO
        if self.config.do_preprocessing:
            solution = self.fixtures.post_process_fixation(solution)
            self.instance = self.fixtures.instance

        solution.costs = solution.compute_costs(self.instance)

        solution.probabilities = solution.compute_probabilities()
        solution.sort_by_cost()

        if self.config.do_postprocessing:
            solution = self.fixtures.postprocess(solution)

        return solution


class QuboSolverClassical(BaseSolver):
    """
    Classical solver for QUBO problems.
    This implementation delegates the classical solving task to the external
    classical solver module (e.g., CPLEX, D-Wave SA, or D-Wave Tabu),
    as selected via the SolverConfig.

    After obtaining the raw solution, postprocessing (e.g., bit-flip local search)
    is applied.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig | None = None):
        super().__init__(instance, config)
        # Optionally, you could instantiate Fixtures here for postprocessing:
        self.fixtures = Fixtures(self.instance, self.config)

    def embedding(self) -> Register:
        # Classical solvers do not require an embedding.
        return  # type: ignore[return-value]

    def drive(self, embedding: Register) -> tuple:
        return  # type: ignore[return-value]

    def solve(self) -> QUBOSolution:
        # 1) try trivial
        if self.config.activate_trivial_solutions:
            trivial = self._trivial_solution()
            if trivial is not None:
                return trivial

        if self.config.do_preprocessing:
            # Apply preprocessing and change the solved QUBO by the reduced one
            self.fixtures.preprocess()
            self.instance = self.fixtures.reduced_qubo
            self.n_fixed_variables_preprocessing = self.fixtures.n_fixed_variables

        classical_solver = get_classical_solver(self.instance, self.config.classical)
        solution = (
            classical_solver.solve()
        )  # This is a reduced solution if pre-procesing is applied

        if self.config.do_preprocessing:
            # Post-process fixations of the preprocessing and restore the original QUBO
            solution = self.fixtures.post_process_fixation(solution)
            self.instance = self.fixtures.instance

        if self.config.do_postprocessing:
            # Apply postprocessing to the raw solution
            solution = self.fixtures.postprocess(solution)

        return solution


class DecomposeQuboSolver(BaseSolver):
    """
    Solver that leverages qubo decomposition techniques to solve
    subproblems and merge subsolutions to propose solutions of
    the original instance. Note, the QUBO instance is seen as a graph where variables
    are vertices, and coefficients represents weighted edges.

    Scope: the decomposition only accepts qubo with positive coefficients off-diagonal coefficients.

    Algorithm:
        1. Initialize global solution and vertices to place.
        2. While we can apply decomposition:
            - Select a random vertex to place using device layout.
            - Apply a geometric search to obtain a set of vertices
                that can be placed on a device to form a subproblem.
            - Solve the subproblem with a solver.
            - Update the global solution and the vertices left to place.
        3. For remaining variables, we solve classically.

    Note, only one bitstring solution is returned.
    """

    def __init__(
        self,
        instance: QUBOInstance,
        config: SolverConfig | None,
        solver_factory: Callable[[QUBOInstance, SolverConfig], BaseSolver],
    ):
        """
        Initialize the DecomposeQuboSolver with the given problem and configuration.

        Args:
            instance (QUBOInstance): The QUBO problem to solve.
            config (SolverConfig): Solver settings including backend and device.
            solver_factory (Callable[[QUBOInstance, SolverConfig], BaseSolver]): solver class
                for subproblems.
        """
        if (
            instance.coefficients[~torch.eye(*instance.coefficients.shape, dtype=torch.bool)] < 0
        ).any():
            raise ValueError("Decomposition does not handle off-diagonal negative coefficients")

        # default is a quantum solver as we apply device-dependent decomposition
        super().__init__(
            QUBOInstance(instance.coefficients),
            config or SolverConfig(use_quantum=True),
        )
        self._solver_factory = solver_factory

        self.fixtures = Fixtures(self.instance, self.config)
        self.backend = self.config.backend
        self.device = self.config.device._device

        self.decomposition_config: DecompositionConfig = (
            self.config.decompose or DecompositionConfig()
        )

        # A cached version of `config` that we're going
        # to use for problems we do not wish to decompose.
        self._config_subproblems = SolverConfig.from_kwargs(
            **self.config.model_dump(exclude={"decompose"})
        )

    def embedding(self) -> Register:
        # This solver doesn't generate an embedding.
        raise NotImplementedError()

    def drive(self, embedding: Register) -> tuple:
        # This solver doesn't generate a drive.
        raise NotImplementedError()

    def solve(self) -> QUBOSolution:
        """
        Execute a solver by decomposing the instance.
        Note, the QUBO instance is seen as a graph where variables
        are vertices, and coefficients represents weighted edges.

        Algorithm:
            1. Initialize global solution and vertices to place.
            2. While we can apply decomposition:
                - Select a random vertex to place using device layout.
                - Apply a geometric search to obtain a set of vertices
                  that can be placed on a device to form a subproblem.
                - Solve the subproblem with a solver.
                - Update the global solution and the vertices left to place.
            3. For remaining variables, we solve classically.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
                Note, only one bitstring solution is returned.
        """

        self.number_iterations = 0
        assert self.instance.size
        if self.instance.size <= self.decomposition_config.decompose_stop_number:
            solver = QuboSolverClassical(
                self.instance,
                SolverConfig(use_quantum=False, decompose=None),
            )
            return solver.solve()

        else:
            from qubosolver.algorithms.decompose import (
                compute_distance_interaction_matrix,
                geometric_search,
                interaction_matrix_from_placed,
                last_target_matrix,
                transfer_edge_values,
                update_global_solution,
                vertices_to_place,
            )

            global_solution = torch.full((self.instance.size,), -1)
            qubo_mat = self.instance.coefficients.clone()
            dist_matrix = compute_distance_interaction_matrix(
                self.device,
                qubo_mat,
                neglecting_inter_distance=self.decomposition_config.neglecting_inter_distance,
                neglecting_max_coefficient=self.decomposition_config.neglecting_max_coefficient,
            )

            # The following dictionary contain vertices to apply the decomposition search
            # where each vertex key maps to other blocking, separated and neighbors vertices
            # and gets updated as we iterate in the decomposition
            dict_vertices_to_place = vertices_to_place(
                dist_matrix,
                qubo_mat,
                separation_threshold=self.decomposition_config.neglecting_inter_distance,
            )

            transfer_edge_values(dict_vertices_to_place, dict(), global_solution, qubo_mat)
            while len(dict_vertices_to_place) > self.decomposition_config.decompose_stop_number:
                # find a first vertex to start the geometric search
                # random works better according to some performed numerics
                first_vertex_search = random.choice(list(dict_vertices_to_place.keys()))
                placed_vertices = geometric_search(
                    qubo_mat,
                    dict_vertices_to_place,
                    first_vertex_search,
                    self.decomposition_config.decompose_threshold,
                    self.device,
                )
                if len(placed_vertices) <= self.decomposition_config.decompose_break_placement:
                    break
                self.number_iterations += 1

                matrix_to_solve, map_index_vertices = interaction_matrix_from_placed(
                    placed_vertices, self.device
                )
                qubo = QUBOInstance(matrix_to_solve)
                subsolver = self._solver_factory(qubo, self._config_subproblems)

                # only one bitstring is kept as per design choice of the
                # decomposition algorithm
                sub_solution = subsolver.solve().bitstrings[0]
                update_global_solution(
                    global_solution=global_solution,
                    sub_solution=sub_solution,
                    mapping=map_index_vertices,
                )

                transfer_edge_values(
                    dict_vertices_to_place,
                    placed_vertices,
                    global_solution,
                    qubo_mat,
                )

            # classical resolution of last matrix
            matrix_to_solve, mapping_target_vertices = last_target_matrix(
                list(dict_vertices_to_place.keys()), qubo_mat
            )
            qubo = QUBOInstance(matrix_to_solve)
            lastsolver = QuboSolverClassical(
                qubo,
                SolverConfig(use_quantum=False, decompose=None),
            )
            sub_solution = lastsolver.solve().bitstrings[0]
            update_global_solution(
                global_solution=global_solution,
                sub_solution=sub_solution,
                mapping=mapping_target_vertices,
            )
            # Probabilities and counts are ignored as we return one solution
            qubosol = QUBOSolution(
                bitstrings=global_solution.unsqueeze(0).to(dtype=torch.float32),
                counts=torch.tensor([1], dtype=torch.int),
                costs=torch.Tensor(),
                probabilities=None,
            )
            qubosol.costs = qubosol.compute_costs(self.instance)
            return qubosol
