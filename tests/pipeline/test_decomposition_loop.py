# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""The decomposition loops documented in the embedding user guide."""

from __future__ import annotations

import numpy as np
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
    numpy_rng,
    solving,
)
from qubosolver.transforms import variable_fixing
from qubosolver.utils._random import manual_seed


def _random_qubo(n: int, density: float, seed: int) -> Instance:
    """The random QUBO of the user guide: negative diagonal, sparse non-negative off-diagonal."""
    gen = numpy_rng(seed)
    upper = np.triu(20 * gen.random((n, n)) * (gen.random((n, n)) < density), k=1)
    Q = upper + upper.T - np.diag(100 * gen.random(n))
    return Instance(Q)


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
    rounds = 0

    while reduced:
        _, variables = embedding.partial_greedy.embed_for_device(reduced, device, rng=rng)
        subinstance = reduced.subinstance(variables)
        subsolution = solving.tabu_search.solve(subinstance)
        variable_fixing.apply_candidate(reduced, subsolution[0], variables, inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1

    assert reduced.size == 0
    solution = variable_fixing.lift(Solution.zeros(0), reduced)

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
    rounds = 0

    while reduced:
        register, variables = embedding.partial_greedy.embed_for_device(reduced, device, rng=rng)
        subinstance = extract_qubo(register, reduced.subinstance(variables))
        subsolution = solving.tabu_search.solve(subinstance)
        variable_fixing.apply_candidate(reduced, subsolution[0], variables, inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1

    assert reduced.size == 0
    solution = variable_fixing.lift(Solution.zeros(0), reduced)

    check.greater(rounds, 0)
    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.usefixtures("restore_rng_state")
def test_classical_loop_with_a_zero_diagonal_coefficient() -> None:
    # Hansen's rule never fixes a tie, so a variable with a zero diagonal coefficient survives
    # `apply_recursively` and reaches the partial embedder.
    size = 20
    rng = manual_seed(699059484)
    upper = np.triu(5 * rng.random((size, size)), k=1)
    Q = upper + upper.T
    np.fill_diagonal(Q, -10.0)
    Q[0, 0] = 0.0
    instance = Instance(Q)
    device = qoolqit.AnalogDeviceWithDMM()

    reduced = variable_fixing.apply_recursively(instance)
    rounds = 0

    while reduced:
        _, variables = embedding.partial_greedy.embed_for_device(reduced, device, rng=rng)
        subinstance = reduced.subinstance(variables)
        subsolution = solving.tabu_search.solve(subinstance)
        variable_fixing.apply_candidate(reduced, subsolution[0], variables, inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1

    assert reduced.size == 0
    solution = variable_fixing.lift(Solution.zeros(0), reduced)

    check.greater(rounds, 0)
    check.equal(solution[0].string[0], "0")
    solution.check_consistency(instance=instance, throw=True)


@pytest.mark.usefixtures("restore_rng_state")
def test_quantum_loop(small_qubo: Instance) -> None:
    rng = manual_seed(0)
    instance = small_qubo
    device = qoolqit.AnalogDeviceWithDMM()
    backend = LocalEmulator(num_shots=100)

    reduced = variable_fixing.apply_recursively(instance)
    rounds = 0

    while reduced:
        register, variables = embedding.partial_greedy.embed_for_device(reduced, device, rng=rng)
        subinstance = reduced.subinstance(variables)
        drive = drive_shaping.proportional_diagonal.build_drive(
            subinstance, register, device=device
        )
        job = backend.run(solving.analog_quantum_sampling.compile(register, drive, device))
        subsolution = Solution.from_results(job.results(), subinstance)
        variable_fixing.apply_candidate(reduced, subsolution[0], variables, inplace=True)
        variable_fixing.apply_recursively(reduced, inplace=True)
        rounds += 1

    assert reduced.size == 0
    solution = variable_fixing.lift(Solution.zeros(0), reduced)

    check.greater(rounds, 0)
    solution.check_consistency(instance=instance, throw=True)
