from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check
import torch

from qoolqit import Drive, Ramp, Register, Constant, DigitalAnalogDevice
from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.qubo_analyzer import QUBOAnalyzer
from qubosolver.config import (
    EmbeddingConfig,
    DriveShapingConfig,
    SolverConfig,
    LocalEmulator,
)
from qubosolver.pipeline.fixtures import (
    Fixtures,
    hansen_fixing,
)
from qubosolver.qubo_types import SolutionStatusType, EmbedderType
from qubosolver.solver import QuboSolver
from qubosolver.pipeline.drive import BaseDriveShaper


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

    sol = QUBOSolution(torch.empty(0, 0), torch.empty(0, 0))

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


def test_reduce_qubo_2() -> None:

    print()
    matrix = torch.tensor(
        [
            [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ],
        dtype=torch.float32,
    )

    qubo = QUBOInstance(matrix)
    config = SolverConfig(do_preprocessing=True)
    fix_class = Fixtures(qubo, config)
    fix_class.preprocess()
    check.equal(fix_class.fixed_var_dict_list, [{0: 0}, {2: 1, 3: 1}])

    # Hardcode solution
    reduced_solution = QUBOSolution(
        bitstrings=torch.tensor([[0, 1]]),
        costs=torch.tensor([0.0]),
    )

    solution = fix_class.post_process_fixation(reduced_solution)
    check.equal(solution.bitstrings.shape, (1, 5))

    bitstring = QUBOAnalyzer.tensor_to_bitstrings(solution.bitstrings.to(torch.int64))[0]
    check.equal(bitstring, "00111")
    check.almost_equal(solution.costs[0], -27.288260)


class SimpleShaper(BaseDriveShaper):
    def generate(
        self,
        register: Register,
    ) -> tuple[Drive, QUBOSolution]:

        # Defining the drive parameters
        omega = 0.01
        delta_i = -0.09
        delta_f = -delta_i
        T = 4000.0

        # Defining the drive
        wf_amp = Constant(T, omega)
        wf_det = Ramp(T, delta_i, delta_f)
        drive = Drive(amplitude=wf_amp, detuning=wf_det)

        return drive, QUBOSolution(torch.Tensor(), torch.Tensor())


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("embedding_method", [EmbedderType.BLADE])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no_pre"])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_quantum_prepostprocessing_2(
    embedding_method: str,
    preprocessing: bool,
    dmm: bool,
) -> None:

    np.random.seed(799)

    Q = np.array(
        [
            [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ]
    )

    instance = QUBOInstance(Q)

    config = SolverConfig(use_quantum=True, do_preprocessing=preprocessing, device=DigitalAnalogDevice())
    config.embedding = EmbeddingConfig(
        embedding_method=embedding_method,
        greedy_spacing=0.1,
        greedy_traps=500,
        min_distance=1.001,
    )

    config.drive_shaping = DriveShapingConfig(drive_shaping_method=SimpleShaper, dmm=dmm)
    config.backend = LocalEmulator(num_shots=50)
    solver = QuboSolver(instance, config)

    solutions = solver.solve()

    analyzer = QUBOAnalyzer([solutions])
    df = analyzer.df
    print(f"\n{df}")

    check.is_true(df["bitstrings"].is_unique)

    expected_solutions = ["00111", "01011"]

    probabilities = [df.set_index("bitstrings")["probs"].get(b, 0.0) for b in expected_solutions]
    check.greater(sum(probabilities), 0.9)

    for b in expected_solutions:
        if b in df["bitstrings"].values:
            cost = df.set_index("bitstrings")["costs"].get(b)
            np.testing.assert_allclose(cost, -27.288260)
