# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import io

import numpy as np
import pytest
import pytest_check as check

from qubosolver import Instance, Solution, bitstrings, transforms
from qubosolver.transforms.negative_bitflip import _has_negative_offdiagonal


def non_bipartisable_negative_qubo_for_bitflip() -> Instance:
    """QUBO where bit flips reduce but do not remove all negative coefficients."""
    return Instance(
        np.asarray(
            [
                [0.0, -2.0, 1.0, 1.0],
                [-2.0, 0.0, -2.0, 1.0],
                [1.0, -2.0, 0.0, -2.0],
                [1.0, 1.0, -2.0, 0.0],
            ]
        )
    )


def non_bipartisable_negative_qubo() -> Instance:
    """QUBO where bit flips reduce but do not remove all negative coefficients."""
    return Instance(
        np.asarray(
            [
                [0.0, -2.0, 1.0, 1.0],
                [-2.0, 0.0, -2.0, 1.0],
                [1.0, -2.0, 0.0, -2.0],
                [1.0, 1.0, -2.0, 0.0],
            ]
        )
    )


def positive_qubo() -> Instance:
    """QUBO with no negative off-diagonal coefficient."""
    return Instance(
        np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
    )


def test_init_preserves_the_parent_instance_type() -> None:
    # __init__ deep-copies parent_instance into _parent_instance; if that
    # parent is itself a transform subclass (e.g. negative_bitflip.Instance),
    # the copy must keep that concrete type, not collapse to the base
    # qubosolver.Instance.
    instance = non_bipartisable_negative_qubo_for_bitflip()
    flipped_parent = transforms.negative_bitflip.apply(instance)

    wrapped = transforms.zeroing.Instance(flipped_parent)

    check.is_instance(wrapped._parent_instance, transforms.negative_bitflip.Instance)


def test_apply_removes_all_negative_offdiagonals() -> None:
    instance = non_bipartisable_negative_qubo()
    check.is_true(_has_negative_offdiagonal(instance.matrix))

    zeroed = transforms.zeroing.apply(instance)

    check.is_instance(zeroed, transforms.zeroing.Instance)
    check.is_false(_has_negative_offdiagonal(zeroed.matrix))


def test_negative_matrix_holds_the_removed_coefficients() -> None:
    instance = non_bipartisable_negative_qubo()
    offdiag = ~np.eye(instance.size, dtype=bool)
    # Capture the negatives before zeroing mutates the matrix in place.
    negative_offdiag_before = offdiag & (instance.matrix < 0)
    removed_values = instance.matrix[negative_offdiag_before].copy()

    zeroed = transforms.zeroing.apply(instance)

    # negative_matrix carries the original values exactly at the zeroed positions...
    np.testing.assert_allclose(
        zeroed.negative_matrix != 0, negative_offdiag_before, rtol=0, strict=True
    )
    np.testing.assert_allclose(
        zeroed.negative_matrix[negative_offdiag_before], removed_values, strict=True
    )
    # ...and those positions are now zero in the reduced matrix.
    check.is_true(np.all(zeroed.matrix[negative_offdiag_before] == 0.0))


def test_zeroed_edges_is_nx2_and_counts_symmetric_pairs_once() -> None:
    instance = non_bipartisable_negative_qubo()
    offdiag = ~np.eye(instance.size, dtype=bool)
    n_negative_offdiag = int((offdiag & (instance.matrix < 0)).sum())

    edges = transforms.zeroing.apply(instance).zeroed_edges

    check.equal(edges.dtype, np.int64)
    # Each symmetric pair counted once: N = (number of negative off-diagonal entries) / 2.
    check.equal(edges.shape, (n_negative_offdiag // 2, 2))
    # Every returned pair is upper-triangular (i < j).
    check.is_true(np.all(edges[:, 0] < edges[:, 1]))


def test_empty_when_nothing_to_zero() -> None:
    instance = positive_qubo()
    zeroed = transforms.zeroing.apply(instance)

    np.testing.assert_allclose(
        zeroed.negative_matrix,
        np.zeros((instance.size, instance.size)),
        strict=True,
    )
    check.is_true(np.all(zeroed.negative_matrix == 0.0))
    np.testing.assert_allclose(
        zeroed.zeroed_edges, np.zeros((0, 2), dtype=np.int64), rtol=0, strict=True
    )


def test_lift_keeps_bitstrings_and_recomputes_costs_against_pre_zeroing() -> None:
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    solution = Solution(
        bitstrings=bitstrings.from_strings(["1111", "0101"]),
        counts=np.asarray([3, 2], dtype=np.int64),
    )

    restored = transforms.zeroing.lift(solution, zeroed)

    # Bitstrings pass through unchanged (zeroing does not permute variables).
    np.testing.assert_allclose(restored.bitstrings, solution.bitstrings, rtol=0, strict=True)
    np.testing.assert_allclose(restored.counts, solution.counts, rtol=0, strict=True)
    # Costs are evaluated against the pre-zeroing matrix, not the zeroed one.
    for sol in restored:
        expected_cost = instance.cost(sol.bitstring)
        check.almost_equal(sol.cost, expected_cost)


def test_lift_sorts_by_pre_zeroing_cost() -> None:
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    # Sorted against the zeroed matrix: "0000" (cost 0) before "1110" (cost 2).
    # Against the pre-zeroing matrix "1110" costs -6, so the order must flip.
    solution = Solution(
        bitstrings=bitstrings.from_strings(["1110", "0000"]),
        counts=np.asarray([1, 1], dtype=np.int64),
    )._update(zeroed)

    restored = transforms.zeroing.lift(solution, zeroed)

    np.testing.assert_allclose(
        restored.bitstrings, bitstrings.from_strings(["1110", "0000"]), rtol=0, strict=True
    )
    np.testing.assert_allclose(restored.costs, np.sort(restored.costs), strict=True)


def test_lift_is_identity_when_nothing_zeroed() -> None:
    zeroed = transforms.zeroing.apply(positive_qubo())

    sol = Solution(bitstrings=bitstrings.from_strings(["101"]), counts=np.asarray([1]))
    restored = transforms.zeroing.lift(sol, zeroed)

    np.testing.assert_allclose(restored.bitstrings, sol.bitstrings, rtol=0, strict=True)


def test_save_load_roundtrips_zeroing_state() -> None:
    # save/load must persist the zeroing-specific state (negative_matrix,
    # parent instance), not just the matrix inherited from the base
    # Instance.save/load.
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    buffer = io.BytesIO()
    zeroed.save(buffer)
    buffer.seek(0)
    loaded = transforms.zeroing.Instance.load(buffer)

    check.is_instance(loaded, transforms.zeroing.Instance)
    np.testing.assert_allclose(loaded.matrix, zeroed.matrix, strict=True)
    np.testing.assert_allclose(loaded.negative_matrix, zeroed.negative_matrix, strict=True)
    np.testing.assert_allclose(
        loaded._parent_instance.matrix, zeroed._parent_instance.matrix, strict=True
    )


def test_load_of_saved_zeroing_instance_can_be_lifted() -> None:
    # A round-tripped instance must remain usable end-to-end: lift() needs
    # _parent_instance and zeroed_edges to be restored correctly.
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    buffer = io.BytesIO()
    zeroed.save(buffer)
    buffer.seek(0)
    loaded = transforms.zeroing.Instance.load(buffer)

    sol = Solution(
        bitstrings=bitstrings.from_strings(["1111", "0101"]),
        counts=np.asarray([3, 2], dtype=np.int64),
    )
    restored = transforms.zeroing.lift(sol, loaded)
    expected = transforms.zeroing.lift(sol, zeroed)

    np.testing.assert_allclose(restored.bitstrings, expected.bitstrings, rtol=0, strict=True)
    np.testing.assert_allclose(restored.costs, expected.costs, strict=True)
    np.testing.assert_allclose(restored.counts, expected.counts, rtol=0, strict=True)
    np.testing.assert_allclose(restored.probabilities, expected.probabilities, strict=True)


def test_save_load_roundtrips_zeroing_over_negative_bitflip_parent() -> None:
    # zeroing.Instance.save/load dispatches on the parent's actual type (via
    # type(...).save / _load_by_tag), so a parent that is itself a richer
    # subclass (e.g. negative_bitflip.Instance) must round-trip with its own
    # state (flips, offset, metrics) intact, not collapse to a plain
    # base Instance.
    instance = non_bipartisable_negative_qubo_for_bitflip()
    flipped_parent = transforms.negative_bitflip.apply(instance)
    zeroed = transforms.zeroing.apply(flipped_parent)

    buffer = io.BytesIO()
    zeroed.save(buffer)
    buffer.seek(0)
    loaded = transforms.zeroing.Instance.load(buffer)

    assert isinstance(loaded._parent_instance, transforms.negative_bitflip.Instance)
    np.testing.assert_allclose(loaded._parent_instance.matrix, flipped_parent.matrix, strict=True)
    np.testing.assert_allclose(
        loaded._parent_instance.flips, flipped_parent.flips, rtol=0, strict=True
    )
    check.equal(loaded._parent_instance._offset, flipped_parent._offset)


def test_base_instance_load_dispatches_to_zeroing_instance() -> None:
    # The generic Instance.load() entry point must dispatch on the tag
    # written by save(), not on the class it's called through, so loading a
    # zeroing.Instance via the base Instance.load() must still return the
    # full subclass (with its zeroing-specific state), not collapse to a
    # plain base Instance.
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    buffer = io.BytesIO()
    zeroed.save(buffer)
    buffer.seek(0)
    loaded = Instance.load(buffer)

    check.is_instance(loaded, transforms.zeroing.Instance)
    np.testing.assert_allclose(loaded.matrix, zeroed.matrix, strict=True)
    check.equal(loaded.zeroing.zeroed_edges.tolist(), zeroed.zeroed_edges.tolist())


def test_zeroing_load_rejects_a_stream_saved_as_a_different_type() -> None:
    # zeroing.Instance.load(f) must reject a stream that was saved as a
    # plain (or otherwise unrelated) Instance, instead of silently returning
    # the wrong type.
    instance = non_bipartisable_negative_qubo()

    buffer = io.BytesIO()
    instance.save(buffer)
    buffer.seek(0)

    with pytest.raises(TypeError):
        transforms.zeroing.Instance.load(buffer)


def test_apply_does_not_alias_the_parent_instance() -> None:
    # _parent_instance must be an independent copy: mutating the caller's
    # instance after apply() must not change what lift() evaluates costs
    # against, or costs silently drift from the true pre-zeroing objective.
    instance = non_bipartisable_negative_qubo()
    zeroed = transforms.zeroing.apply(instance)

    instance._matrix.fill(0.0)

    sol = Solution(bitstrings=bitstrings.from_strings(["1111"]), counts=np.asarray([1]))
    restored = transforms.zeroing.lift(sol, zeroed)

    check.not_equal(restored[0].cost, 0.0)
    check.almost_equal(restored[0].cost, zeroed._parent_instance.cost(sol[0].bitstring))
