from __future__ import annotations

import numpy as np
import pytest
import pytest_check as check

from qubosolver import Instance, matrix
from qubosolver.embedding import blade


def test_empty_embedding() -> None:
    instance = Instance(matrix.zeros(0))
    with pytest.raises(ValueError, match="empty instance"):
        blade.embed(instance)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_single_atom_embedding(value: float) -> None:
    instance = Instance(matrix.zeros(1).fill_(value))
    register = blade.embed(instance)
    check.equal(len(register), 1)


def captured_starting_positions(
    monkeypatch: pytest.MonkeyPatch, instance: Instance, config: blade.Config
) -> np.ndarray | None:
    captured = {}
    real_blade = blade.Blade

    def fake_blade(config: blade.BladeConfig) -> blade.Blade:
        captured["config"] = config
        return real_blade(config)

    monkeypatch.setattr(blade, "Blade", fake_blade)
    blade.embed(instance, config=config)
    return captured["config"].starting_positions


QUBO = [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]]


def test_initialize_with_mds_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(matrix.as_tensor(QUBO))
    positions = captured_starting_positions(monkeypatch, instance, blade.Config(steps_per_round=10))
    assert positions is not None
    check.equal(positions.shape, (3, 2))


def test_initialize_without_mds(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(matrix.as_tensor(QUBO))
    config = blade.Config(steps_per_round=10, initialize_with_mds=False)
    check.is_none(captured_starting_positions(monkeypatch, instance, config))


def test_explicit_starting_positions_override_mds(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(matrix.as_tensor(QUBO))
    starting_positions = np.arange(6.0).reshape(3, 2)
    config = blade.Config(steps_per_round=10, starting_positions=starting_positions)
    positions = captured_starting_positions(monkeypatch, instance, config)
    assert positions is not None
    check.is_true(np.array_equal(positions, starting_positions))


def test_mds_skipped_without_positive_coupling(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(matrix.as_tensor(np.diag([1.0, 2.0, 3.0])))
    config = blade.Config(steps_per_round=10)
    check.is_none(captured_starting_positions(monkeypatch, instance, config))


def test_tuned_defaults() -> None:
    config = blade.Config()
    check.equal(config.dimensions, (6, 5, 4, 3, 2, 2, 2))
    check.is_true(config.initialize_with_mds)
    for progress in (0.0, 0.5, 1.0):
        check.equal(config.compute_weight_relative_threshold(progress), 0.1)
        check.equal(config.compute_regulation_cursor(progress), 0.5)
