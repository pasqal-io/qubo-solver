from __future__ import annotations

from collections import Counter

import torch

# Import the classical solver factory from our classical_solver module.
from qoolqit._solvers import get_backend

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.classical_solver import get_classical_solver
from qubosolver.config import SolverConfig
from qubosolver.pipeline import (
    BaseSolver,
    Fixtures,
    Pulse,
    Register,
    get_embedder,
    get_pulse_shaper,
)


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
            if config.use_quantum:
                self._solver = QuboSolverQuantum(instance, config)
            else:
                self._solver = QuboSolverClassical(instance, config)

        self.n_fixed_variables_preprocessing = 0

    def embedding(self) -> Register:
        return self._solver.embedding()

    def pulse(self, embedding: Register) -> tuple:
        return self._solver.pulse(embedding)

    def solve(self) -> QUBOSolution:
        return self._solver.solve()


class QuboSolverQuantum(BaseSolver):
    """
    Quantum solver that orchestrates the solving of a QUBO problem using
    embedding, pulse shaping, and quantum execution pipelines.
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
        self.backend = get_backend(self.config.backend_config)
        self.embedder = get_embedder(self.instance, self.config, self.backend)
        self.pulse_shaper = get_pulse_shaper(self.instance, self.config, self.backend)

        self._register: Register | None = None
        self._pulse: Pulse | None = None

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

    def pulse(self, embedding: Register) -> tuple:
        """
        Generate the pulse sequence based on the given embedding.

        Args:
            embedding (Register): The embedded register layout.

        Returns:
            tuple:
                A tuple of
                    - Pulse: Pulse schedule for quantum execution.
                    - QUBOSolution: Initial solution of generated from pulse shaper

        """
        pulse, qubo_solution = self.pulse_shaper.generate(embedding, self.instance)

        self._pulse = pulse
        return pulse, qubo_solution

    def solve(self) -> QUBOSolution:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute, postprocess.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
        """
        # 1) try trivial and verify size
        trivial = self._trivial_solution()
        if trivial is not None and self.config.activate_trivial_solutions:
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

        pulse, qubo_solution = self.pulse(embedding)

        bitstrings, counts, _, _ = (
            qubo_solution.bitstrings,
            qubo_solution.counts,
            qubo_solution.probabilities,
            qubo_solution.costs,
        )
        if (
            len(bitstrings) == 0 and qubo_solution.counts is None
        ) or self.config.pulse_shaping.re_execute_opt_pulse:
            bitstrings, counts = self.execute(pulse, embedding)

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

    def pulse(self, embedding: Register) -> tuple:
        return  # type: ignore[return-value]

    def solve(self) -> QUBOSolution:
        # 1) try trivial
        trivial = self._trivial_solution()
        if trivial is not None:
            return trivial

        # 2) else delegate to quantum or classical solver
        # Delegate the solving to the classical solver module.
        # Convert the SolverConfig instance to a dictionary.
        config_dict = self.config.__dict__

        if self.config.do_preprocessing:
            # Apply preprocessing and change the solved QUBO by the reduced one
            self.fixtures.preprocess()
            self.instance = self.fixtures.reduced_qubo
            self.n_fixed_variables_preprocessing = self.fixtures.n_fixed_variables

        classical_solver = get_classical_solver(self.instance, config_dict)
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
