from __future__ import annotations

import random
import numpy as np
import pytest
import pytest_check as check
import torch
from copy import deepcopy
from typing import Literal

from qoolqit import DigitalAnalogDevice
from qubosolver import (
    Instance,
    Solution,
    transforms,
    Solver,
    matrix,
    bitstring,
    bitstrings,
    vector,
    LocalEmulator,
    SolverConfig,
    QuantumSolvingConfig,
    ClassicalSolvingConfig,
    EmbeddingConfig,
    DriveShapingConfig,
)
from qubosolver.utils import analysis


def test_apply_full_and_post_process_fixation() -> None:

    Q = matrix.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
    )

    full_qubo = Instance(Q)
    reduced_qubo = transforms.variable_fixing.apply_recursively(full_qubo)

    assert reduced_qubo._fixed_indices == [{0: 1, 3: 1}, {0: 0, 1: 0}]
    assert reduced_qubo.n_fixed_indices == 4

    reduced_solution = Solution(bitstrings.zeros(0, 0), vector.zeros(0))

    sol_reconstructed = transforms.variable_fixing.lift(reduced_solution, reduced_qubo)

    assert isinstance(sol_reconstructed, Solution)

    val_red = int(sol_reconstructed.costs[0])

    assert val_red == -153


def test_hansen_fixing() -> None:
    matrix_reducible = matrix.tensor([[-10, 1], [1, -10]])

    qubo_reducible = Instance(matrix_reducible)

    fixed_var = transforms.variable_fixing.hansen_fixing(qubo_reducible)

    assert isinstance(fixed_var, dict)

    assert fixed_var == {0: 1, 1: 1}

    matrix_not_reducible = matrix.tensor([[-1, 10], [10, -1]])

    qubo_reducible = Instance(matrix_not_reducible)

    empty_fixed_var = transforms.variable_fixing.hansen_fixing(qubo_reducible)

    assert empty_fixed_var == {}


def test_reduce_qubo() -> None:

    Q = matrix.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
    )
    qubo = Instance(Q)
    reduced_qubo = deepcopy(qubo)

    reduced_qubo = transforms.variable_fixing._reduce_qubo(qubo, {0: 1, 3: 1})

    expected_coefficients = matrix.tensor([[22, 20], [20, 6]])
    torch.testing.assert_close(reduced_qubo.matrix, expected_coefficients)

    non_reduced_qubo = transforms.variable_fixing._reduce_qubo(qubo, {})
    torch.testing.assert_close(non_reduced_qubo.matrix, qubo.matrix)


def test_apply_rule() -> None:

    Q = matrix.tensor(
        [[-98, 2, 13, 1], [2, -12, 20, 15], [13, 20, -34, 7], [1, 15, 7, -57]],
    )
    qubo = Instance(Q)
    reduced_qubo = transforms.variable_fixing.apply(
        qubo, [transforms.variable_fixing.hansen_fixing]
    )

    expected_coefficients = matrix.tensor([[22, 20], [20, 6]])
    torch.testing.assert_close(reduced_qubo.matrix, expected_coefficients)


def test_quantum_preprocessing_falls_back_to_zeroing_when_bitflip_is_not_enough(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A non-bipartisable QUBO: bit-flip preprocessing cannot remove every
    negative off-diagonal coefficient, so the quantum solver must zero the
    rest out automatically (and log it) before embedding.
    """
    Q = matrix.tensor(
        [
            [0.0, -2.0, 1.0, 1.0],
            [-2.0, 0.0, -2.0, 1.0],
            [1.0, -2.0, 0.0, -2.0],
            [1.0, 1.0, -2.0, 0.0],
        ]
    )
    instance = Instance(Q)
    flipped = transforms.negative_bitflip.apply(instance)
    check.is_true(transforms.negative_bitflip._has_negative_offdiagonal(flipped.matrix))

    config = SolverConfig(
        solving=QuantumSolvingConfig(),
        preprocessing=True,
        activate_trivial_solutions=False,
        postprocessing=False,
    )
    solver = Solver(instance, config)

    with caplog.at_level("INFO", logger="qubosolver.solver.solver"):
        solution = solver.solve()

    check.is_true(
        any("zeroing the remainder" in record.message for record in caplog.records),
        "expected the automatic-zeroing fallback to log a message",
    )
    check.equal(len(solution[0].string), instance.size)
    # Costs in the returned solution must be evaluated against the true,
    # original QUBO, not the zeroed approximation used internally.
    for sol in solution:
        check.almost_equal(sol.cost, instance.cost(sol.bitstring))


def test_quantum_preprocessing(qubo_instance_for_preprocessing: Instance) -> None:
    """
    Test instance using quantum with preprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=QuantumSolvingConfig(), preprocessing=True, postprocessing=False
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_quantum_postprocessing(qubo_instance_for_preprocessing: Instance) -> None:
    """
    Test instance using quantum with postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=QuantumSolvingConfig(), preprocessing=False, postprocessing=True
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_quantum_prepostprocessing(
    qubo_instance_for_preprocessing: Instance,
) -> None:
    """
    Test instance using quantum with both preprocessing and postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=QuantumSolvingConfig(), preprocessing=True, postprocessing=True
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_preprocessing(qubo_instance_for_preprocessing: Instance) -> None:
    """
    Test instance using classical with preprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=ClassicalSolvingConfig(), preprocessing=True, postprocessing=False
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_postprocessing(
    qubo_instance_for_preprocessing: Instance,
) -> None:
    """
    Test instance using classical with postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=ClassicalSolvingConfig(), preprocessing=False, postprocessing=True
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_classical_prepostprocessing(
    qubo_instance_for_preprocessing: Instance,
) -> None:
    """
    Test instance using classical with preprocessing and postprocessing.
    """
    quantum_preprocessing_config = SolverConfig(
        solving=ClassicalSolvingConfig(), preprocessing=True, postprocessing=True
    )
    solver = Solver(qubo_instance_for_preprocessing, quantum_preprocessing_config)
    solution = solver.solve()
    assert len(solution.bitstrings[0]) == qubo_instance_for_preprocessing.size


def test_reduce_qubo_2() -> None:

    Q = matrix.tensor(
        [
            [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ],
    )

    qubo = Instance(Q)

    reduced_qubo = transforms.variable_fixing.apply_recursively(qubo)

    check.equal(reduced_qubo._fixed_indices, [{0: 0}, {2: 1, 3: 1}])

    # Hardcode solution
    reduced_solution = Solution(
        bitstrings=bitstrings.tensor([[0, 1]]),
        costs=vector.tensor([0.0]),
    )

    solution = transforms.variable_fixing.lift(reduced_solution, reduced_qubo)
    check.equal(solution.bitstrings.shape, (1, 5))

    bitstring_ = bitstring.to_string(solution.bitstrings[0])
    check.equal(bitstring_, "00111")
    check.almost_equal(solution.costs[0], -27.288260)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("embedding_method", ["blade"])
@pytest.mark.parametrize("preprocessing", [True, False], ids=["pre", "no_pre"])
@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_quantum_prepostprocessing_2(
    embedding_method: Literal["blade"],
    preprocessing: bool,
    dmm: bool,
) -> None:

    seed = 799
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    Q = matrix.tensor(
        [
            [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ]
    )

    instance = Instance(Q)

    config = SolverConfig(
        solving=QuantumSolvingConfig(
            EmbeddingConfig(
                algorithm=embedding_method,
                greedy_layout_traps=500,
            ),
            drive_shaping=DriveShapingConfig(dmm=dmm),
            device=DigitalAnalogDevice(),
            backend=LocalEmulator(num_shots=50),
        ),
        preprocessing=preprocessing,
    )
    solver = Solver(instance, config)

    solutions = solver.solve()

    df = analysis.to_dataframe([solutions])
    print(f"\n{df}")

    check.is_true(df["bitstrings"].is_unique)

    expected_solutions = ["00111", "01011"]

    probabilities = [df.set_index("bitstrings")["probs"].get(b, 0.0) for b in expected_solutions]
    check.greater(sum(probabilities), 0.9)

    for b in expected_solutions:
        if b in df["bitstrings"].values:
            cost = df.set_index("bitstrings")["costs"].get(b)
            np.testing.assert_allclose(cost, -27.288260)
