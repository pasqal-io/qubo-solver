"""The decomposition loops documented in the embedding user guide."""

from __future__ import annotations

import pytest
import pytest_check as check
import qoolqit
import torch

from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    drive_shaping,
    embedding,
    extract_qubo,
    matrix,
    solving,
)
from qubosolver.transforms import variable_fixing
from qubosolver.utils._random import manual_seed


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
    solution.check_consistency(instance=instance, throw=True)


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
    solution.check_consistency(instance=instance, throw=True)


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
    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.duration(30)
@pytest.mark.usefixtures("restore_rng_state")
def test_user_guide_quantum_example() -> None:
    manual_seed(2)
    n = 12
    upper = (20 * torch.rand(n, n) * (torch.rand(n, n) < 0.5)).triu(1)
    Q = upper + upper.T - torch.diag(100 * torch.rand(n))
    instance = Instance(matrix.as_tensor(Q))
    device = qoolqit.AnalogDeviceWithDMM()
    backend = LocalEmulator(num_shots=1000)

    reduced_instance = variable_fixing.apply_recursively(instance)
    reduced_solution = solving.trivial_solution_search.solve(reduced_instance)
    while not reduced_solution:
        register, variables = embedding.geometric.embed_for_device(reduced_instance, device)
        sub_instance = reduced_instance.subinstance(variables)
        drive = drive_shaping.local_energy_scale.build_drive(sub_instance, register, device=device)
        program = solving.analog_quantum_sampling.compile(register, drive, device)
        job = backend.run(program)
        sub_solution = Solution.from_results(job.results(), sub_instance)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced_instance)
        variable_fixing.apply(reduced_instance, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced_instance, inplace=True)
        reduced_solution = solving.trivial_solution_search.solve(reduced_instance)
    solution = variable_fixing.lift(reduced_solution, reduced_instance)

    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.usefixtures("restore_rng_state")
def test_user_guide_classical_example() -> None:
    manual_seed(10)
    n = 50
    upper = (20 * torch.rand(n, n) * (torch.rand(n, n) < 0.3)).triu(1)
    Q = upper + upper.T - torch.diag(100 * torch.rand(n))
    instance = Instance(matrix.as_tensor(Q))
    device = qoolqit.AnalogDeviceWithDMM()

    reduced_instance = variable_fixing.apply_recursively(instance)
    reduced_solution = solving.trivial_solution_search.solve(reduced_instance)
    while not reduced_solution:
        _, variables = embedding.geometric.embed_for_device(reduced_instance, device)
        sub_instance = reduced_instance.subinstance(variables)
        sub_solution = solving.brute_force.solve(sub_instance)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced_instance)
        variable_fixing.apply(reduced_instance, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced_instance, inplace=True)
        reduced_solution = solving.trivial_solution_search.solve(reduced_instance)
    solution = variable_fixing.lift(reduced_solution, reduced_instance)

    solution.check_consistency(instance=instance, throw=True)
