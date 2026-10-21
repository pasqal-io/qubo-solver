"""The decomposition loops documented in the embedding user guide."""

from __future__ import annotations

import pytest
import pytest_check as check
import qoolqit

from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    drive_shaping,
    embedding,
    extract_qubo,
    solving,
)
from qubosolver.transforms import variable_fixing
from qubosolver.utils._random import manual_seed


def check_complete_solution(solution: Solution, instance: Instance) -> None:
    check.equal(len(solution), 1)
    check.equal(len(solution[0].bitstring), instance.size)
    check.almost_equal(solution[0].cost, instance.cost(solution[0].bitstring), rel=1e-5)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classical_loop_on_exact_subproblems(decomposable_qubo: Instance, seed: int) -> None:
    manual_seed(seed)
    instance = decomposable_qubo
    device = qoolqit.AnalogDeviceWithDMM()
    stop, min_atoms = 15, 3

    reduced = variable_fixing.apply_recursively(instance)
    rounds = 0
    while reduced.size > stop:
        _, variables = embedding.geometric.embed_for_device(reduced, device)
        if len(variables) <= min_atoms:
            break
        sub = reduced.subinstance(variables)
        sub_solution = solving.tabu_search.solve(sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1
    solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

    check.greater(rounds, 0)
    check.equal(len(reduced.fixed_variables) + reduced.size, instance.size)
    check_complete_solution(solution, instance)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classical_loop_on_register_derived_subproblems(
    decomposable_qubo: Instance, seed: int
) -> None:
    manual_seed(seed)
    instance = decomposable_qubo
    device = qoolqit.AnalogDeviceWithDMM()
    stop, min_atoms = 15, 3

    reduced = variable_fixing.apply_recursively(instance)
    rounds = 0
    while reduced.size > stop:
        register, variables = embedding.geometric.embed_for_device(reduced, device)
        if len(variables) <= min_atoms:
            break
        exact = reduced.subinstance(variables)
        sub = extract_qubo(register, exact)
        sub_solution = solving.tabu_search.solve(sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1
    solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

    check.greater(rounds, 0)
    check_complete_solution(solution, instance)


@pytest.mark.duration(60)
@pytest.mark.usefixtures("restore_rng_state")
def test_quantum_loop(small_decomposable_qubo: Instance) -> None:
    manual_seed(0)
    instance = small_decomposable_qubo
    device = qoolqit.AnalogDeviceWithDMM()
    backend = LocalEmulator(num_shots=100)
    stop, min_atoms = 4, 1

    reduced = variable_fixing.apply_recursively(instance)
    rounds = 0
    while reduced.size > stop:
        register, variables = embedding.geometric.embed_for_device(reduced, device)
        if len(variables) <= min_atoms:
            break
        sub = reduced.subinstance(variables)
        drive = drive_shaping.proportional_diagonal.build_drive(sub, register, device=device)
        job = backend.run(solving.analog_quantum_sampling.compile(register, drive, device))
        sub_solution = Solution.from_results(job.results(), sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1
    solution = variable_fixing.lift(solving.tabu_search.solve(reduced), reduced)

    check.greater(rounds, 0)
    check_complete_solution(solution, instance)
