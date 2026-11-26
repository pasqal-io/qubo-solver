# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytest_check as check
import qoolqit

from qubosolver import Instance
from qubosolver.embedding import blade

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_empty_embedding() -> None:
    instance = Instance(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="empty instance"):
        blade.embed(instance)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_single_atom_embedding(value: float) -> None:
    instance = Instance(np.full((1, 1), value, dtype=np.float64))
    register = blade.embed(instance)
    check.equal(len(register), 1)


def test_single_atom_embedding_for_device() -> None:
    instance = Instance(np.asarray([[-1.0]]))
    register = blade.embed_for_device(instance, qoolqit.AnalogDeviceWithDMM())
    check.equal(len(register), 1)


def test_embed_rejects_diagonal_instance() -> None:
    instance = Instance(np.asarray(np.diag([-1.0, -2.0, -3.0]), dtype=np.float64))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        blade.embed(instance)


def test_embed_for_device_rejects_diagonal_instance() -> None:
    instance = Instance(np.asarray(np.diag([-1.0, -2.0, -3.0]), dtype=np.float64))
    with pytest.raises(ValueError, match="trivial_solution_search"):
        blade.embed_for_device(instance, qoolqit.AnalogDeviceWithDMM())


def captured_starting_positions(
    monkeypatch: pytest.MonkeyPatch, instance: Instance, config: blade.Config
) -> NDArray[np.float64] | None:
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
    instance = Instance(np.asarray(QUBO, dtype=np.float64))
    positions = captured_starting_positions(monkeypatch, instance, blade.Config(steps_per_round=10))
    assert positions is not None
    check.equal(positions.shape, (3, 2))


def test_initialize_without_mds(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(np.asarray(QUBO, dtype=np.float64))
    config = blade.Config(steps_per_round=10, initialize_with_mds=False)
    check.is_none(captured_starting_positions(monkeypatch, instance, config))


def test_explicit_starting_positions_override_mds(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = Instance(np.asarray(QUBO, dtype=np.float64))
    starting_positions = np.arange(6.0).reshape(3, 2)
    config = blade.Config(steps_per_round=10, starting_positions=starting_positions)
    positions = captured_starting_positions(monkeypatch, instance, config)
    assert positions is not None
    check.is_true(np.array_equal(positions, starting_positions))


def test_tuned_defaults() -> None:
    config = blade.Config()
    check.equal(config.dimensions, (6, 5, 4, 3, 2, 2, 2))
    check.is_true(config.initialize_with_mds)
    for progress in (0.0, 0.5, 1.0):
        check.equal(config.compute_weight_relative_threshold(progress), 0.1)
        check.equal(config.compute_regulation_cursor(progress), 0.5)


def test_negative_off_diagonal_raises() -> None:
    instance = Instance(np.asarray([[1.0, -1.0], [-1.0, 1.0]]))
    with pytest.raises(ValueError, match="negative off-diagonal"):
        blade.embed(instance)
