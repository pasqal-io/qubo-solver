# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

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
    torch_rng,
)
from qubosolver.transforms import variable_fixing
from qubosolver.utils._random import manual_seed


def _random_qubo(n: int, density: float, seed: int) -> Instance:
    """The random QUBO of the user guide: negative diagonal, sparse non-negative off-diagonal."""
    gen = torch_rng(seed)
    upper = (
        20 * torch.rand(n, n, generator=gen) * (torch.rand(n, n, generator=gen) < density)
    ).triu(1)
    Q = upper + upper.T - torch.diag(100 * torch.rand(n, generator=gen))
    return Instance(matrix.as_tensor(Q))


@pytest.fixture
def random_qubo() -> Instance:
    return _random_qubo(50, 0.3, 10)


@pytest.fixture
def small_random_qubo() -> Instance:
    return _random_qubo(12, 0.5, 2)


@pytest.fixture(params=["decomposable_qubo", "random_qubo"])
def qubo(request: pytest.FixtureRequest) -> Instance:
    instance: Instance = request.getfixturevalue(request.param)
    return instance


@pytest.fixture(params=["small_decomposable_qubo", "small_random_qubo"])
def small_qubo(request: pytest.FixtureRequest) -> Instance:
    instance: Instance = request.getfixturevalue(request.param)
    return instance


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classical_loop_on_exact_subproblems(qubo: Instance, seed: int) -> None:
    rng = manual_seed(seed)
    instance = qubo
    device = qoolqit.AnalogDeviceWithDMM()

    reduced = variable_fixing.apply_recursively(instance)
    reduced_solution = solving.trivial_solution_search.solve(reduced)
    rounds = 0

    while not reduced_solution:
        _, variables = embedding.geometric.embed_for_device(reduced, device, rng=rng)
        sub = reduced.subinstance(variables)
        sub_solution = solving.tabu_search.solve(sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        reduced_solution = solving.trivial_solution_search.solve(reduced)
        rounds += 1

    solution = variable_fixing.lift(reduced_solution, reduced)

    check.greater(rounds, 0)
    check.equal(len(reduced.fixed_variables) + reduced.size, instance.size)
    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classical_loop_on_register_derived_subproblems(qubo: Instance, seed: int) -> None:
    rng = manual_seed(seed)
    instance = qubo
    device = qoolqit.AnalogDeviceWithDMM()

    reduced = variable_fixing.apply_recursively(instance)
    reduced_solution = solving.trivial_solution_search.solve(reduced)
    rounds = 0

    while not reduced_solution:
        register, variables = embedding.geometric.embed_for_device(reduced, device, rng=rng)
        exact = reduced.subinstance(variables)
        sub = extract_qubo(register, exact)
        sub_solution = solving.tabu_search.solve(sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        reduced_solution = solving.trivial_solution_search.solve(reduced)
        rounds += 1

    solution = variable_fixing.lift(reduced_solution, reduced)

    check.greater(rounds, 0)
    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.usefixtures("restore_rng_state")
def test_quantum_loop(small_qubo: Instance) -> None:
    rng = manual_seed(0)
    instance = small_qubo
    device = qoolqit.AnalogDeviceWithDMM()
    backend = LocalEmulator(num_shots=100)

    reduced = variable_fixing.apply_recursively(instance)
    reduced_solution = solving.trivial_solution_search.solve(reduced)
    rounds = 0

    while not reduced_solution:
        register, variables = embedding.geometric.embed_for_device(reduced, device, rng=rng)
        sub = reduced.subinstance(variables)
        drive = drive_shaping.proportional_diagonal.build_drive(sub, register, device=device)
        job = backend.run(solving.analog_quantum_sampling.compile(register, drive, device))
        sub_solution = Solution.from_results(job.results(), sub)
        rule = variable_fixing.from_candidate(sub_solution[0], variables, reduced)
        variable_fixing.apply(reduced, (rule,), inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        reduced_solution = solving.trivial_solution_search.solve(reduced)
        rounds += 1

    solution = variable_fixing.lift(reduced_solution, reduced)

    check.greater(rounds, 0)
    solution.check_consistency(instance=instance, throw=True)
