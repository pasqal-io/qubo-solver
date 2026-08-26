from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check
import random
import torch
import qoolqit

from qubosolver import (
    Instance,
    Solution,
    SingleSolution,
    solvers,
    embedding,
    drive_shaping,
    torch_rng,
    extract_qubo,
)

from qubosolver.utils import analysis

from qubos import QUBOS


def manual_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return torch_rng(seed)


def gather_optimal_solutions(solutions: Solution) -> list[SingleSolution]:
    min_cost = solutions[0].cost
    return [d for d in solutions if np.allclose(d.cost, min_cost)]


def check_solution(
    solution: Solution,
    qubo: Instance,
    expect_optimality: bool = True,
) -> float:

    # Solutions are not duplicated
    check.equal(solution.bitstrings.unique(dim=0).shape[0], len(solution))

    print(f"\n{analysis.to_dataframe([solution])}")

    optimal_solutions = gather_optimal_solutions(solution)
    check.is_not(optimal_solutions, [])

    min_cost = optimal_solutions[0].cost

    print(f"\nMinimum cost: {min_cost}")
    print(f"All optimal bitstrings: {[s.string for s in optimal_solutions]}")
    print(f"Number of optimal solutions: {len(optimal_solutions)}\n")

    if not expect_optimality:
        return 0.0

    expected_optimal_solutions = gather_optimal_solutions(
        solvers.brute_force.solve(qubo, max_bitstrings=-1)
    )
    check.almost_equal(min_cost, expected_optimal_solutions[0].cost)
    expected_optimal_bitstrings = [s.string for s in expected_optimal_solutions]
    for s in optimal_solutions:
        check.is_in(s.string, expected_optimal_bitstrings)

    cumulated_probability = sum(s.probability for s in optimal_solutions)
    return cumulated_probability


def _relative_norm(diff_norm: float, ref_norm: float) -> float:
    if np.isclose(ref_norm, 0.0):
        print("Reference norm is ~0, printing absolute norm instead")
        return diff_norm
    return diff_norm / ref_norm


def check_hamiltonian_qubo(
    register: qoolqit.Register, drive: qoolqit.Drive, qubo: Instance
) -> tuple[float, float]:
    extracted = extract_qubo(register, drive)
    with np.printoptions(precision=2, suppress=True):
        print(f"\nExtracted QUBO matrix:\n{extracted.matrix.numpy()}\n")
        print(f"Expected QUBO matrix:\n{qubo.matrix.numpy()}\n")

    diff = extracted.matrix - qubo.matrix
    diagonal_diff_norm = torch.diag(diff).norm().item()
    off_diagonal_diff_norm = diff.fill_diagonal_(0.0).norm().item()

    diagonal_norm = torch.diag(qubo.matrix).norm().item()
    off_diagonal_norm = qubo.matrix.clone().fill_diagonal_(0.0).norm().item()

    rel_diagonal_error = _relative_norm(diagonal_diff_norm, diagonal_norm)
    rel_offdiagonal_error = _relative_norm(off_diagonal_diff_norm, off_diagonal_norm)

    print(f"Relative diagonal diff norm: {rel_diagonal_error:.2f}")
    print(f"Relative off-diagonal diff norm: {rel_offdiagonal_error:.2f}")

    return rel_diagonal_error, rel_offdiagonal_error


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [16844214, 650], ids=[f"seed{i}" for i in range(2)])
@pytest.mark.parametrize(
    "qubo_id, expected_diag_error, expected_offdiag_error, expected_optimum_prob",
    [
        (0, 0.001, 0.1, 0.8),
        (1, 1.2, 0.1, 0.001),
        (2, 1.6, 0.1, 0.01),
        (3, 1.6, 0.2, 0.5),
        (4, 1.6, 0.3, 0.001),
        (5, 1.6, 0.5, 0.1),
        (6, 1.6, 0.3, 0.7),
        (7, 1.6, 0.1, 0.2),
        (8, 1.6, 0.1, 0.6),
        (9, 1.6, 0.1, 0.001),
        (10, 1.0, 0.5, 0.01),
        (11, 1.6, 0.5, 0.01),
    ],
    ids=[f"qubo{i}" for i in range(12)],
)
def test_quantum_solve_blade_proportional_diagonal(
    qubo_id: int,
    expected_diag_error: float,
    expected_offdiag_error: float,
    expected_optimum_prob: float,
    seed: int,
) -> None:

    manual_seed(seed)

    instance = QUBOS[qubo_id]

    device = qoolqit.AnalogDeviceWithDMM()
    emulator = qoolqit.execution.LocalEmulator()

    blade_config = embedding.blade.Config(device=device)
    register = embedding.blade.embed(instance, config=blade_config)

    drive = drive_shaping.proportional_diagonal.build_drive(
        instance,
        register,
        device=device,
        dmm=True,
    )

    job = solvers.analog_quantum_sampling.solve(register, drive, emulator, device)
    solution = Solution.from_results(job.results(), instance)
    solution._compute_costs(instance.matrix)._sort_by_cost()._compute_probabilities()

    optimum_prob = check_solution(solution, instance)
    check.greater_equal(optimum_prob, expected_optimum_prob)

    diag_error, offdiag_error = check_hamiltonian_qubo(register, drive, instance)
    check.less_equal(diag_error, expected_diag_error)
    check.less_equal(offdiag_error, expected_offdiag_error)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [42, 271828], ids=[f"seed{i}" for i in range(2)])
@pytest.mark.parametrize(
    "qubo_id, expected_diag_error, expected_offdiag_error, expected_optimum_prob",
    [
        (0, 0.001, 0.5, 0.2),
        (1, 1.2, 0.5, 0.001),
        (2, 1.6, 0.5, 0.01),
        (3, 1.6, 0.5, 0.5),
        (4, 1.6, 0.8, 0.001),
        (5, 1.6, 0.8, 0.01),
        (6, 1.6, 0.5, 0.2),
        (7, 1.6, 0.5, 0.1),
        (8, 1.6, 0.8, 0.01),
        (9, 1.6, 0.5, 0.001),
        (10, 1.0, 0.8, 0.05),
        (11, 1.6, 0.8, 0.0),
    ],
    ids=[f"qubo{i}" for i in range(12)],
)
def test_quantum_solve_greedy_proportional_diagonal(
    qubo_id: int,
    expected_diag_error: float,
    expected_offdiag_error: float,
    expected_optimum_prob: float,
    seed: int,
) -> None:

    manual_seed(seed)

    instance = QUBOS[qubo_id]

    device = qoolqit.AnalogDeviceWithDMM()
    emulator = qoolqit.execution.LocalEmulator()

    greedy_config = embedding.greedy_layout.Config(traps=100)
    register = embedding.greedy_layout.embed(instance, device=device, config=greedy_config)

    drive = drive_shaping.proportional_diagonal.build_drive(
        instance,
        register,
        device=device,
        dmm=True,
    )

    job = solvers.analog_quantum_sampling.solve(register, drive, emulator, device)
    solution = Solution.from_results(job.results(), instance)
    solution._compute_costs(instance.matrix)._sort_by_cost()._compute_probabilities()

    expect_optimality = expected_optimum_prob > 0.0
    optimum_prob = check_solution(solution, instance, expect_optimality)
    check.greater_equal(optimum_prob, expected_optimum_prob)

    diag_error, offdiag_error = check_hamiltonian_qubo(register, drive, instance)
    check.less_equal(diag_error, expected_diag_error)
    check.less_equal(offdiag_error, expected_offdiag_error)
