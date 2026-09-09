from __future__ import annotations

import io

import pytest_check as check
import torch

from qubosolver import Instance, Solution, transforms, bitstrings, matrix, solving, vectori
from qubosolver.transforms.variable_fixing import hansen_fixing


def bipartisable_negative_qubo_for_bitflip() -> Instance:
    """QUBO where bit flips can remove all negative off-diagonal coefficients."""
    return Instance(
        matrix.tensor(
            [
                [0.0, -2.0, -1.0, 1.0],
                [-2.0, 0.0, 1.0, -1.0],
                [-1.0, 1.0, 0.0, -2.0],
                [1.0, -1.0, -2.0, 0.0],
            ]
        )
    )


def fixable_qubo() -> Instance:
    """QUBO where recursive Hansen fixing stabilizes with 2 of 5 variables left."""
    return Instance(
        matrix.tensor(
            [
                [0.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
                [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
                [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
                [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
                [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
            ]
        )
    )


def unfixable_qubo() -> Instance:
    """QUBO where Hansen's criterion cannot fix any variable."""
    return Instance(
        matrix.tensor(
            [
                [1.0, -3.0, 3.0],
                [-3.0, 1.0, -3.0],
                [3.0, -3.0, 1.0],
            ]
        )
    )


def test_hansen_fixing_identifies_fixed_variables() -> None:
    instance = fixable_qubo()
    fixed = hansen_fixing(instance)

    check.equal(fixed, {0: 0})


def test_hansen_fixing_is_empty_when_nothing_can_be_fixed() -> None:
    fixed = hansen_fixing(unfixable_qubo())
    check.equal(fixed, {})


def test_init_preserves_the_parent_instance_type() -> None:
    # __init__ deep-copies parent_instance into _parent_instance; if that
    # parent is itself a transform subclass (e.g. negative_bitflip.Instance),
    # the copy must keep that concrete type, not collapse to the base
    # qubosolver.Instance.
    instance = bipartisable_negative_qubo_for_bitflip()
    flipped_parent = transforms.negative_bitflip.apply(instance)

    wrapped = transforms.variable_fixing.Instance(flipped_parent)

    check.is_instance(wrapped._parent_instance, transforms.negative_bitflip.Instance)


def test_apply_reduces_matrix_and_records_fixed_indices() -> None:
    instance = fixable_qubo()
    expected_fixed = hansen_fixing(instance)

    reduced = transforms.variable_fixing.apply(instance)

    check.is_instance(reduced, transforms.variable_fixing.Instance)
    check.equal(reduced.fixed_indices, [expected_fixed])
    check.equal(reduced.n_fixed_indices, len(expected_fixed))
    check.equal(reduced.size, instance.size - len(expected_fixed))


def test_apply_is_noop_when_nothing_can_be_fixed() -> None:
    instance = unfixable_qubo()
    reduced = transforms.variable_fixing.apply(instance)

    check.equal(reduced.size, instance.size)
    check.equal(reduced.n_fixed_indices, 0)
    torch.testing.assert_close(reduced.matrix, instance.matrix)


def test_apply_recursively_fixes_until_stable() -> None:
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)
    n_fixed_indices = reduced.n_fixed_indices

    # A further in-place pass finds nothing new: the recursion already ran
    # to a fixed point.
    transforms.variable_fixing.apply(reduced, inplace=True)
    check.equal(reduced.n_fixed_indices, n_fixed_indices)


def test_lift_reinserts_fixed_variables_and_recomputes_costs() -> None:
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)

    reduced_solution = solving.brute_force.solve(reduced)
    restored_solution = transforms.variable_fixing.lift(reduced_solution, reduced)

    for restored in restored_solution:
        check.equal(len(restored.bitstring), instance.size)
        expected_cost = instance.cost(restored.bitstring)
        check.almost_equal(restored.cost, expected_cost)


def test_lift_is_a_copy_when_nothing_fixed() -> None:
    instance = unfixable_qubo()
    reduced = transforms.variable_fixing.apply(instance)

    sol = Solution(bitstrings=bitstrings.from_strings(["101"]), counts=vectori.tensor([1]))
    sol._update(instance)
    restored = transforms.variable_fixing.lift(sol, reduced)

    torch.testing.assert_close(restored.bitstrings, sol.bitstrings)


def test_apply_does_not_alias_the_parent_instance() -> None:
    # _parent_instance must be an independent copy: mutating the caller's
    # instance after apply() must not change what lift() evaluates costs
    # against.
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)

    instance._matrix.fill_(0.0)

    reduced_solution = solving.brute_force.solve(reduced)
    restored_solution = transforms.variable_fixing.lift(reduced_solution, reduced)

    for restored in restored_solution:
        check.not_equal(restored.cost, 0.0)


def test_save_load_roundtrips_variable_fixing_state() -> None:
    # save/load must persist the fixing-specific state (fixed_indices,
    # parent instance), not just the matrix inherited from the base
    # Instance.save/load.
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)

    buffer = io.BytesIO()
    reduced.save(buffer)
    buffer.seek(0)
    loaded = transforms.variable_fixing.Instance.load(buffer)

    check.is_instance(loaded, transforms.variable_fixing.Instance)
    torch.testing.assert_close(loaded.matrix, reduced.matrix)
    check.equal(loaded.fixed_indices, reduced.fixed_indices)
    torch.testing.assert_close(loaded._parent_instance.matrix, reduced._parent_instance.matrix)


def test_base_instance_load_dispatches_to_variable_fixing_instance() -> None:
    # The generic Instance.load() entry point must dispatch on the tag
    # written by save(), not on the class it's called through, so loading a
    # variable_fixing.Instance via the base Instance.load() must still return
    # the full subclass (with its fixing-specific state), not collapse to a
    # plain base Instance.
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)

    buffer = io.BytesIO()
    reduced.save(buffer)
    buffer.seek(0)
    loaded = Instance.load(buffer)

    check.is_instance(loaded, transforms.variable_fixing.Instance)
    torch.testing.assert_close(loaded.matrix, reduced.matrix)
    check.equal(loaded.variable_fixing.fixed_indices, reduced.fixed_indices)
    torch.testing.assert_close(
        loaded.variable_fixing._parent_instance.matrix, reduced._parent_instance.matrix
    )


def test_load_of_saved_variable_fixing_instance_can_be_lifted() -> None:
    # A round-tripped instance must remain usable end-to-end: lift() needs
    # _parent_instance and fixed_indices to be restored correctly.
    instance = fixable_qubo()
    reduced = transforms.variable_fixing.apply_recursively(instance)
    reduced_solution = solving.brute_force.solve(reduced)

    buffer = io.BytesIO()
    reduced.save(buffer)
    buffer.seek(0)
    loaded = transforms.variable_fixing.Instance.load(buffer)

    restored = transforms.variable_fixing.lift(reduced_solution, loaded)
    expected = transforms.variable_fixing.lift(reduced_solution, reduced)

    torch.testing.assert_close(restored.bitstrings, expected.bitstrings)
    torch.testing.assert_close(restored.costs, expected.costs)
    torch.testing.assert_close(restored.counts, expected.counts)
    torch.testing.assert_close(restored.probabilities, expected.probabilities)


def test_save_load_roundtrips_variable_fixing_over_negative_bitflip_parent() -> None:
    # variable_fixing.Instance.save/load dispatches on the parent's actual
    # type (via type(...).save / _load_by_tag), so a parent that is itself a
    # richer subclass (e.g. negative_bitflip.Instance) must round-trip with
    # its own state (flips, status, offset, metrics) intact, not collapse to
    # a plain base Instance.
    instance = bipartisable_negative_qubo_for_bitflip()
    flipped_parent = transforms.negative_bitflip.apply(instance)
    reduced = transforms.variable_fixing.apply(flipped_parent)

    buffer = io.BytesIO()
    reduced.save(buffer)
    buffer.seek(0)
    loaded = transforms.variable_fixing.Instance.load(buffer)

    assert isinstance(loaded._parent_instance, transforms.negative_bitflip.Instance)
    torch.testing.assert_close(loaded._parent_instance.matrix, flipped_parent.matrix)
    torch.testing.assert_close(loaded._parent_instance.flips, flipped_parent.flips)
    check.equal(loaded._parent_instance.status, flipped_parent.status)
    check.equal(loaded._parent_instance.offset, flipped_parent.offset)
