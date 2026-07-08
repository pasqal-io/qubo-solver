from __future__ import annotations
from typing import Literal
import pytest
import torch

from qubosolver import QUBOInstance, QUBOSolution
from qubosolver.config import BitFlipPreprocessingConfig, SolverConfig
from qubosolver.pipeline.bitflip_preprocessing import has_negative_offdiagonal
from qubosolver.solver import QuboSolverClassical, QuboSolverQuantum


def bipartisable_negative_qubo() -> torch.Tensor:
    """QUBO where bit flips can remove all negative off-diagonal coefficients."""
    return torch.tensor(
        [
            [0.0, -2.0, -1.0, 1.0],
            [-2.0, 0.0, 1.0, -1.0],
            [-1.0, 1.0, 0.0, -2.0],
            [1.0, -1.0, -2.0, 0.0],
        ]
    )


def non_bipartisable_negative_qubo() -> torch.Tensor:
    """QUBO where bit flips reduce but do not remove all negative coefficients."""
    return torch.tensor(
        [
            [0.0, -2.0, 1.0, 1.0],
            [-2.0, 0.0, -2.0, 1.0],
            [1.0, -2.0, 0.0, -2.0],
            [1.0, 1.0, -2.0, 0.0],
        ]
    )


def bitflip_config(
    *,
    negative_handling: Literal["error", "zeroing"] = "error",
    do_preprocessing: bool = True,
) -> SolverConfig:
    return SolverConfig(
        use_quantum=False,
        do_preprocessing=do_preprocessing,
        activate_trivial_solutions=False,
        negative_handling=negative_handling,
        bitflip_preprocessing=BitFlipPreprocessingConfig(
            enabled=True,
            time_limit_s=5.0,
        ),
    )


def quantum_bitflip_config(
    *,
    negative_handling: Literal["error", "zeroing"] = "error",
    do_preprocessing: bool = True,
) -> SolverConfig:
    return SolverConfig(
        use_quantum=True,
        do_preprocessing=do_preprocessing,
        activate_trivial_solutions=False,
        negative_handling=negative_handling,
        bitflip_preprocessing=BitFlipPreprocessingConfig(
            enabled=True,
            time_limit_s=5.0,
        ),
    )


def test_has_negative_offdiagonal_detects_negative_interactions() -> None:
    Q = torch.tensor(
        [
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    )

    assert has_negative_offdiagonal(Q)


def test_has_negative_offdiagonal_ignores_negative_diagonal() -> None:
    Q = torch.tensor(
        [
            [-3.0, 1.0],
            [1.0, -2.0],
        ]
    )

    assert not has_negative_offdiagonal(Q)


def test_bitflip_preprocessing_removes_all_negative_coefficients() -> None:
    Q = bipartisable_negative_qubo()
    solver = QuboSolverClassical(QUBOInstance(Q), bitflip_config())

    solver.preprocess()

    assert solver.fixtures.bitflip_applied
    assert not solver.fixtures.zeroing_applied
    assert not has_negative_offdiagonal(solver.instance.coefficients)


def test_bitflip_preprocessing_can_leave_negative_coefficients() -> None:
    Q = non_bipartisable_negative_qubo()
    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(negative_handling="error"),
    )

    solver.preprocess()

    assert solver.fixtures.bitflip_applied
    assert not solver.fixtures.zeroing_applied
    assert has_negative_offdiagonal(solver.instance.coefficients)


def test_zeroing_handles_remaining_negative_coefficients_after_bitflip() -> None:
    Q = non_bipartisable_negative_qubo()
    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(negative_handling="zeroing"),
    )

    solver.preprocess()

    assert solver.fixtures.bitflip_applied
    assert solver.fixtures.zeroing_applied
    assert not has_negative_offdiagonal(solver.instance.coefficients)


def test_error_handling_does_not_zero_remaining_negative_coefficients() -> None:
    Q = non_bipartisable_negative_qubo()
    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(negative_handling="error"),
    )

    solver.preprocess()

    assert solver.fixtures.bitflip_applied
    assert not solver.fixtures.zeroing_applied
    assert has_negative_offdiagonal(solver.instance.coefficients)


def test_original_qubo_is_unchanged_after_preprocessing() -> None:
    Q = non_bipartisable_negative_qubo()
    original_Q = Q.clone()

    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(negative_handling="zeroing"),
    )

    solver.preprocess()

    assert torch.equal(solver.fixtures.instance.coefficients, original_Q)


def test_basesolver_uses_preprocessed_qubo_even_without_size_reduction() -> None:
    Q = non_bipartisable_negative_qubo()

    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(negative_handling="zeroing"),
    )

    solver.preprocess()

    assert solver.fixtures.n_fixed_variables == 0
    assert solver.fixtures.zeroing_applied
    assert torch.equal(solver.instance.coefficients, solver.fixtures.reduced_qubo.coefficients)
    assert not has_negative_offdiagonal(solver.instance.coefficients)


def test_bitflip_postprocess_restores_original_bitstrings() -> None:
    Q = bipartisable_negative_qubo()
    solver = QuboSolverClassical(QUBOInstance(Q), bitflip_config())

    solver.preprocess()

    raw_solution = QUBOSolution(
        bitstrings=torch.tensor([[0, 0, 0, 0]], dtype=torch.float32),
        costs=torch.Tensor(),
        counts=torch.tensor([1]),
        probabilities=None,
    )

    restored = solver.post_process_fixation(raw_solution)

    assert solver.fixtures.bitflip_vector is not None
    assert torch.equal(restored.bitstrings[0], solver.fixtures.bitflip_vector.float())

    expected_cost = QUBOInstance(Q).evaluate_solution(restored.bitstrings[0].tolist())
    assert torch.isclose(
        restored.costs[0],
        torch.tensor(expected_cost, dtype=torch.float32),
    )


def test_no_preprocessing_when_do_preprocessing_is_false() -> None:
    Q = non_bipartisable_negative_qubo()

    solver = QuboSolverClassical(
        QUBOInstance(Q),
        bitflip_config(
            negative_handling="zeroing",
            do_preprocessing=False,
        ),
    )

    solver.preprocess()

    assert not solver.fixtures.bitflip_applied
    assert not solver.fixtures.zeroing_applied
    assert torch.equal(solver.instance.coefficients, Q)


def test_quantum_solver_rejects_negative_coefficients_without_bitflip_preprocessing() -> None:
    Q = non_bipartisable_negative_qubo()

    config = SolverConfig(
        use_quantum=True,
        do_preprocessing=False,
        activate_trivial_solutions=False,
    )

    with pytest.raises(ValueError, match="off-diagonal negative coefficients"):
        QuboSolverQuantum(QUBOInstance(Q), config)


def test_quantum_solver_constructor_accepts_negative_coefficients_with_bitflip_enabled() -> None:
    Q = non_bipartisable_negative_qubo()

    solver = QuboSolverQuantum(
        QUBOInstance(Q),
        quantum_bitflip_config(negative_handling="error"),
    )

    assert isinstance(solver, QuboSolverQuantum)


def test_quantum_solver_raises_after_preprocessing_if_negative_coefficients_remain() -> None:
    Q = non_bipartisable_negative_qubo()

    solver = QuboSolverQuantum(
        QUBOInstance(Q),
        quantum_bitflip_config(negative_handling="error"),
    )

    with pytest.raises(ValueError, match="Preprocessing did not remove all negative coefficients"):
        solver.solve()


def test_quantum_solver_preprocess_passes_when_zeroing_removes_remaining_negatives() -> None:
    Q = non_bipartisable_negative_qubo()

    solver = QuboSolverQuantum(
        QUBOInstance(Q),
        quantum_bitflip_config(negative_handling="zeroing"),
    )

    solver.preprocess()

    assert solver.fixtures.bitflip_applied
    assert solver.fixtures.zeroing_applied
    assert not has_negative_offdiagonal(solver.instance.coefficients)
