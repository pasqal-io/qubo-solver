from __future__ import annotations

import io
import itertools
from typing import Any

import pytest
import pytest_check as check
import torch

from qubosolver import (
    Instance,
    transforms,
    bitstrings,
    matrix,
    bitstring,
    solving,
)
from qubosolver.utils._costs import quadratic_cost
from qubosolver.transforms.negative_bitflip import (
    _apply_bitflips,
    _transform_qubo_with_bitflips,
    _has_negative_offdiagonal,
    _solve_bitflip_preprocessing_glpk,
)


def bipartisable_negative_qubo() -> tuple[Instance, Instance]:
    """QUBO where bit flips can remove all negative off-diagonal coefficients."""
    instance = Instance(
        matrix.tensor(
            [
                [0.0, -2.0, -1.0, 1.0],
                [-2.0, 0.0, 1.0, -1.0],
                [-1.0, 1.0, 0.0, -2.0],
                [1.0, -1.0, -2.0, 0.0],
            ]
        )
    )
    flipped_instance = Instance(
        matrix.tensor(
            [
                [-2.0, 2.0, 1.0, 1.0],
                [2.0, -6.0, 1.0, 1.0],
                [1.0, 1.0, -6.0, 2.0],
                [1.0, 1.0, 2.0, -2.0],
            ]
        )
    )
    return instance, flipped_instance


def non_bipartisable_negative_qubo() -> tuple[Instance, Instance]:
    """QUBO where bit flips reduce but do not remove all negative coefficients."""
    instance = Instance(
        matrix.tensor(
            [
                [0.0, -2.0, 1.0, 1.0],
                [-2.0, 0.0, -2.0, 1.0],
                [1.0, -2.0, 0.0, -2.0],
                [1.0, 1.0, -2.0, 0.0],
            ]
        )
    )
    flipped_instance = Instance(
        matrix.tensor(
            [
                [-2.0, 2.0, 1.0, -1.0],
                [2.0, -8.0, 2.0, 1.0],
                [1.0, 2.0, -2.0, 2.0],
                [-1.0, 1.0, 2.0, -2.0],
            ]
        )
    )
    return instance, flipped_instance


def test_has_negative_offdiagonal_detects_negative_interactions() -> None:
    Q = matrix.tensor(
        [
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    )

    check.is_true(_has_negative_offdiagonal(Q))


def test_has_negative_offdiagonal_ignores_negative_diagonal() -> None:
    Q = matrix.tensor(
        [
            [-3.0, 1.0],
            [1.0, -2.0],
        ]
    )

    check.is_false(_has_negative_offdiagonal(Q))


def test_bitflip_preprocessing_removes_all_negative_coefficients() -> None:
    instance, expected_flipped_instance = bipartisable_negative_qubo()
    flipped_instance = transforms.negative_bitflip.apply(instance)

    check.is_instance(flipped_instance, transforms.negative_bitflip.Instance)
    torch.testing.assert_close(flipped_instance.matrix, expected_flipped_instance.matrix)
    check.is_true(flipped_instance.flips.any())
    check.is_false(_has_negative_offdiagonal(flipped_instance.matrix))


def test_bitflip_preprocessing_can_leave_negative_coefficients() -> None:
    instance, expected_flipped_instance = non_bipartisable_negative_qubo()
    flipped_instance = transforms.negative_bitflip.apply(instance)

    check.is_instance(flipped_instance, transforms.negative_bitflip.Instance)
    torch.testing.assert_close(flipped_instance.matrix, expected_flipped_instance.matrix)
    check.is_true(flipped_instance.flips.any())
    check.is_true(_has_negative_offdiagonal(flipped_instance.matrix))


def test_bitflip_apply_is_noop_without_negative_coefficients() -> None:
    instance = Instance(matrix.tensor([[1.0, 2.0], [2.0, 1.0]]))

    flipped_instance = transforms.negative_bitflip.apply(instance)

    check.is_false(flipped_instance.flips.any())
    check.equal(flipped_instance.offset, 0.0)
    torch.testing.assert_close(flipped_instance.matrix, instance.matrix)


def test_bitflip_lift_restores_original_variables() -> None:
    instance, _ = bipartisable_negative_qubo()
    solution = solving.brute_force.solve(instance)

    flipped_instance = transforms.negative_bitflip.apply(instance)
    flipped_solution = solving.brute_force.solve(flipped_instance)

    restored = transforms.negative_bitflip.lift(flipped_solution, flipped_instance)

    torch.testing.assert_close(restored[0].bitstring, solution[0].bitstring)
    check.almost_equal(restored[0].cost, solution[0].cost)


def test_transform_qubo_with_bitflips_preserves_objective() -> None:
    # The whole scheme rests on this identity: for every assignment y of the
    # flipped QUBO, cost_original(x) == cost_flipped(y) + offset, where x is y
    # with the flipped variables complemented (x_i = 1 - y_i when flips_i = 1).
    Q, _ = bipartisable_negative_qubo()
    n = Q.size
    flips = bitstring.from_string("1001")

    Q_flipped, offset = _transform_qubo_with_bitflips(Q.matrix, flips)

    for bits in itertools.product([0, 1], repeat=n):
        y = bitstring.tensor(bits)
        x = torch.abs(y - flips)  # undo the flips: y -> x
        cost_original = quadratic_cost(x, Q.matrix)
        cost_flipped = quadratic_cost(y, Q_flipped) + offset
        check.almost_equal(cost_flipped, cost_original)


def test_apply_bitflips_is_batched_xor_and_involutive() -> None:
    flips = bitstring.from_string("1001")
    # Multiple rows with non-trivial overlap with the flip vector: some bits set
    # where flips are set (cancel), some not (stay/appear).
    batch = bitstrings.from_strings(["1100", "0101", "1011"])
    expected = bitstrings.from_strings(["0101", "1100", "0010"])

    flipped = _apply_bitflips(batch, flips)

    # Per-row XOR against the flip vector (broadcast over the batch).
    torch.testing.assert_close(flipped, expected)
    # Applying the same flips twice restores the original batch (involution).
    torch.testing.assert_close(_apply_bitflips(flipped, flips), batch)


def test_apply_bitflips_empty_batch_is_noop() -> None:
    flips = bitstring.from_string("1001")
    empty = bitstrings.zeros(0, 4)

    torch.testing.assert_close(_apply_bitflips(empty, flips), empty)


def test_metrics_reflect_the_original_matrix_not_the_flipped_one() -> None:
    # neg_count_before/neg_weight_before must be computed against the original
    # QUBO, not the already-flipped one, or "before" and "after" both describe
    # the flipped matrix and reduction percentages become meaningless.
    instance, _ = non_bipartisable_negative_qubo()
    flipped_instance = transforms.negative_bitflip.apply(instance)

    n = instance.size
    expected_before_count = 0
    expected_before_weight = 0.0
    expected_after_count = 0
    expected_after_weight = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            q_before = instance.matrix[i, j].item()
            if q_before < 0:
                expected_before_count += 1
                expected_before_weight += abs(q_before)
            q_after = flipped_instance.matrix[i, j].item()
            if q_after < 0:
                expected_after_count += 1
                expected_after_weight += abs(q_after)

    check.equal(flipped_instance.metrics["neg_count_before"], expected_before_count)
    check.almost_equal(flipped_instance.metrics["neg_weight_before"], expected_before_weight)
    check.equal(flipped_instance.metrics["neg_count_after"], expected_after_count)
    check.almost_equal(flipped_instance.metrics["neg_weight_after"], expected_after_weight)
    # For this fixture, bit-flip preprocessing only reduces, it does not remove.
    check.greater(expected_before_count, expected_after_count)


def test_bitflip_lift_restores_original_variables_for_a_batch() -> None:
    instance, _ = bipartisable_negative_qubo()
    solution = solving.brute_force.solve(instance, max_bitstrings=4)
    solution._sort_by_cost()

    flipped_instance = transforms.negative_bitflip.apply(instance)
    flipped_solution = solving.brute_force.solve(flipped_instance, max_bitstrings=4)

    restored = transforms.negative_bitflip.lift(flipped_solution, flipped_instance)
    restored._sort_by_cost()

    torch.testing.assert_close(restored.costs, solution.costs)
    torch.testing.assert_close(restored[0].bitstring, solution[0].bitstring)

    # Every restored bitstring must be exactly the flip-undo of its
    # corresponding flipped-QUBO bitstring, and evaluate to the same cost on
    # the original matrix.
    for i in range(len(restored)):
        x = torch.abs(flipped_solution[i].bitstring - flipped_instance.flips)
        torch.testing.assert_close(restored[i].bitstring, x)
        check.almost_equal(restored[i].cost, instance.cost(restored[i].bitstring))


def test_glpk_infeasible_or_error_falls_back_to_noop_flips() -> None:
    # A vanishing time limit forces GLPK to hit its time limit before finding
    # any feasible solution; apply() must degrade to a safe no-op rather than
    # raise or return garbage flips.
    instance, _ = non_bipartisable_negative_qubo()

    flips, _, status = _solve_bitflip_preprocessing_glpk(instance.matrix, time_limit_s=0.0)

    check.is_in(status, ("TIME_LIMIT_NO_SOLUTION", "TIME_LIMIT_FEASIBLE", "OPTIMAL", "FEASIBLE"))
    check.equal(flips.shape[0], instance.size)


def test_glpk_time_limit_warns_when_falling_back(caplog: pytest.LogCaptureFixture) -> None:
    # When GLPK can't reach a feasible solution in time, apply() must degrade
    # to no-op flips *and* surface a warning, so the failure is observable.
    instance, _ = non_bipartisable_negative_qubo()

    with caplog.at_level("WARNING", logger="qubosolver.transforms.negative_bitflip"):
        flips, _, status = _solve_bitflip_preprocessing_glpk(instance.matrix, time_limit_s=0.0)

    if status in ("TIME_LIMIT_NO_SOLUTION", "GLPK_ERROR_0"):
        check.is_false(flips.any())
        check.is_true(len(caplog.records) >= 1)


def test_apply_rejects_flips_that_increase_negative_weight(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression test: GLPK can return flips that are individually "optimal"
    # for its internal objective but still leave *more* negative weight than
    # doing nothing (see issue #237). apply() must detect and reject this,
    # falling back to a safe no-op rather than handing back a worse matrix.
    instance, _ = non_bipartisable_negative_qubo()

    # These flips are a genuine regression on this fixture: negative weight
    # goes from 6.0 (no-op) to 7.0 (flipped).
    worse_flips = bitstring.from_string("1100")

    def _fake_solve(*args: Any, **kwargs: Any) -> tuple[Any, float, str]:
        return worse_flips, 0.0, "OPTIMAL"

    monkeypatch.setattr(
        "qubosolver.transforms.negative_bitflip._solve_bitflip_preprocessing_glpk",
        _fake_solve,
    )

    with caplog.at_level("WARNING", logger="qubosolver.transforms.negative_bitflip"):
        flipped_instance = transforms.negative_bitflip.apply(instance)

    check.equal(flipped_instance.status, "REJECTED_WORSE_THAN_NOOP")
    check.is_false(flipped_instance.flips.any())
    check.is_nan(flipped_instance.metrics["objective_value"])
    torch.testing.assert_close(flipped_instance.matrix, instance.matrix)
    check.equal(
        flipped_instance.metrics["neg_weight_after"],
        flipped_instance.metrics["neg_weight_before"],
    )
    check.is_true(len(caplog.records) >= 1)


def test_apply_keeps_flips_that_reduce_negative_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sanity check for the new guard: a genuinely-improving flip vector must
    # still be kept and must not be flagged as a regression.
    instance, _ = non_bipartisable_negative_qubo()
    good_flips = bitstring.from_string("0100")
    expected_matrix, _ = _transform_qubo_with_bitflips(instance.matrix, good_flips)

    def _fake_solve(*args: Any, **kwargs: Any) -> tuple[Any, float, str]:
        return good_flips, 0.0, "OPTIMAL"

    monkeypatch.setattr(
        "qubosolver.transforms.negative_bitflip._solve_bitflip_preprocessing_glpk",
        _fake_solve,
    )

    flipped_instance = transforms.negative_bitflip.apply(instance)

    check.equal(flipped_instance.status, "OPTIMAL")
    check.is_true(flipped_instance.flips.any())
    torch.testing.assert_close(flipped_instance.matrix, expected_matrix)


def test_save_load_roundtrips_bitflip_state() -> None:
    # save/load must persist the bit-flip specific state (flips, metrics,
    # status, offset, parent instance), not just the matrix inherited from
    # the base Instance.save/load.
    instance, _ = non_bipartisable_negative_qubo()
    flipped_instance = transforms.negative_bitflip.apply(instance)

    buffer = io.BytesIO()
    transforms.negative_bitflip.Instance.save(buffer, flipped_instance)
    buffer.seek(0)
    loaded_instance = transforms.negative_bitflip.Instance.load(buffer)

    check.is_instance(loaded_instance, transforms.negative_bitflip.Instance)
    torch.testing.assert_close(loaded_instance.matrix, flipped_instance.matrix)
    torch.testing.assert_close(loaded_instance.flips, flipped_instance.flips)
    check.equal(loaded_instance.status, flipped_instance.status)
    check.equal(loaded_instance.offset, flipped_instance.offset)
    check.equal(loaded_instance.metrics, flipped_instance.metrics)
    torch.testing.assert_close(
        loaded_instance._parent_instance.matrix, flipped_instance._parent_instance.matrix
    )


def test_load_of_saved_bitflip_instance_can_be_lifted() -> None:
    # A round-tripped instance must remain usable end-to-end: lift() needs
    # _parent_instance and flips to be restored correctly.
    instance, _ = bipartisable_negative_qubo()
    flipped_instance = transforms.negative_bitflip.apply(instance)
    flipped_solution = solving.brute_force.solve(flipped_instance)

    buffer = io.BytesIO()
    transforms.negative_bitflip.Instance.save(buffer, flipped_instance)
    buffer.seek(0)
    loaded_instance = transforms.negative_bitflip.Instance.load(buffer)

    restored_solution = transforms.negative_bitflip.lift(flipped_solution, loaded_instance)
    expected_solution = transforms.negative_bitflip.lift(flipped_solution, flipped_instance)

    for restored, expected in zip(restored_solution, expected_solution):
        check.equal(restored.string, expected.string)
        check.equal(restored.count, expected.count)
        check.almost_equal(restored.cost, expected.cost)
        check.almost_equal(restored.probability, expected.probability)


def test_glpk_solve_survives_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # If GLPK itself raises during the solve (e.g. a binding error), the
    # function must still return a usable (no-op) result instead of
    # propagating the exception, since apply() has no outer safety net.
    instance, _ = non_bipartisable_negative_qubo()

    import swiglpk as glp

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated GLPK failure")

    monkeypatch.setattr(glp, "glp_intopt", _raise)

    flips, objective_value, status = _solve_bitflip_preprocessing_glpk(instance.matrix)

    check.equal(status, "FAIL")
    check.is_false(flips.any())
    check.is_nan(objective_value)
