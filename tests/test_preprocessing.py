from __future__ import annotations

import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.config import SolverConfig
from qubosolver.pipeline.fixtures import (
    Fixtures,
    dwave_roof_duality_fixing,
    hansen_fixing,
)
from qubosolver.qubo_types import SolutionStatusType
from qubosolver.solver import QuboSolver


def test_apply_full_and_post_process_fixation() -> None:

    matrix = torch.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
        dtype=torch.int32,
    )

    qubo = QUBOInstance(matrix)

    config = SolverConfig(do_preprocessing=True)
    fix_class = Fixtures(qubo, config)
    fix_class.apply_full_fixation_exhaust()

    assert fix_class.fixed_var_dict_list == [{0: 1, 3: 1}, {0: 0, 1: 0}]
    assert fix_class.n_fixed_variables > 0

    assert isinstance(fix_class.reduced_qubo, QUBOInstance)

    sol = QUBOSolution(torch.empty(0, 0), 0)

    sol_reconstructed = fix_class.post_process_fixation(sol)

    assert isinstance(sol_reconstructed, QUBOSolution)

    val_red = int(sol_reconstructed.costs[0])

    assert val_red == -153


def test_hansen_fixing() -> None:
    matrix_reducible = torch.tensor([[-10, 1], [1, -10]], dtype=torch.int32)

    qubo_reducible = QUBOInstance(matrix_reducible)

    fixed_var = hansen_fixing(qubo_reducible)

    assert isinstance(fixed_var, dict)

    assert fixed_var == {0: 1, 1: 1}

    matrix_not_reducible = torch.tensor([[-1, 10], [10, -1]], dtype=torch.int32)

    qubo_reducible = QUBOInstance(matrix_not_reducible)

    empty_fixed_var = hansen_fixing(qubo_reducible)

    assert empty_fixed_var == {}


def test_dwave_roof_duality_fixing() -> None:
    matrix_reducible = torch.tensor([[-10, 1], [1, -10]], dtype=torch.int32)

    qubo_reducible = QUBOInstance(matrix_reducible)

    fixed_var = dwave_roof_duality_fixing(qubo_reducible)

    assert isinstance(fixed_var, dict)

    assert fixed_var == {0: 1, 1: 1}

    matrix_not_reducible = torch.tensor([[-1, 10], [10, -1]], dtype=torch.int32)

    qubo_reducible = QUBOInstance(matrix_not_reducible)

    empty_fixed_var = hansen_fixing(qubo_reducible)

    assert empty_fixed_var == {}


def test_reduce_qubo() -> None:

    matrix = torch.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
        dtype=torch.int32,
    )

    qubo = QUBOInstance(matrix)
    config = SolverConfig(do_preprocessing=True)
    fix_class = Fixtures(qubo, config)

    fix_class.reduce_qubo({0: 1, 3: 1})

    assert torch.equal(
        fix_class.reduced_qubo.coefficients,
        torch.tensor([[22, 20], [20, 6]], dtype=torch.int32),
    )

    fix_class_not_reduced = Fixtures(qubo, config)

    fix_class_not_reduced.reduce_qubo({})

    assert torch.equal(
        fix_class_not_reduced.reduced_qubo.coefficients,
        torch.tensor(
            [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
            dtype=torch.int32,
        ),
    )


def test_apply_rule() -> None:
    matrix = torch.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
        dtype=torch.int32,
    )

    qubo = QUBOInstance(matrix)
    config = SolverConfig(do_preprocessing=True)
    fix_class = Fixtures(qubo, config)

    fix_class.apply_rule(hansen_fixing)

    assert torch.equal(
        fix_class.reduced_qubo.coefficients,
        torch.tensor([[22, 20], [20, 6]], dtype=torch.int32),
    )


def test_classical_unprocessed(qubo_instance_for_preprocessing: QUBOInstance) -> None:
    """
    Test unprocessed instance using quantum
    """
    classical_unprocessed_config = SolverConfig(
        use_quantum=False, do_preprocessing=False, do_postprocessing=False
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, classical_unprocessed_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.UNPROCESSED


def test_quantum_unprocessed(qubo_instance_for_preprocessing: QUBOInstance) -> None:
    """
    Test unprocessed instance using quantum
    """
    quantum_unprocessed_config = SolverConfig(
        use_quantum=True, do_preprocessing=False, do_postprocessing=False
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_unprocessed_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.UNPROCESSED


def test_quantum_preprocessing(qubo_instance_for_preprocessing: QUBOInstance) -> None:
    """
    Test instance using quantum with preprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=True, do_preprocessing=True, do_postprocessing=False
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.PREPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_quantum_postprocessing(qubo_instance_for_preprocessing: QUBOInstance) -> None:
    """
    Test instance using quantum with postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=True, do_preprocessing=False, do_postprocessing=True
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.POSTPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_quantum_prepostprocessing(
    qubo_instance_for_preprocessing: QUBOInstance,
) -> None:
    """
    Test instance using quantum with both preprocessing and postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=True, do_preprocessing=True, do_postprocessing=True
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.PREPOSTPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_preprocessing(qubo_instance_for_preprocessing: QUBOInstance) -> None:
    """
    Test instance using classical with preprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=False, do_preprocessing=True, do_postprocessing=False
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.PREPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_postprocessing(
    qubo_instance_for_preprocessing: QUBOInstance,
) -> None:
    """
    Test instance using classical with postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=False, do_preprocessing=False, do_postprocessing=True
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.POSTPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_prepostprocessing(
    qubo_instance_for_preprocessing: QUBOInstance,
) -> None:
    """
    Test instance using classical with preprocessing and postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        use_quantum=False, do_preprocessing=True, do_postprocessing=True
    )
    solver = QuboSolver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert solution.solution_status == SolutionStatusType.PREPOSTPROCESSED
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size
