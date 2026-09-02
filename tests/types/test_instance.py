# tests/test_qubo_instance.py
from __future__ import annotations

import io
import os
from pathlib import Path
import numpy as np
import pytest
import pytest_check as check
import torch

from qubosolver import Instance, Solver, solving, matrix, transforms, SolverConfig, ClassicalSolvingConfig, QuantumSolvingConfig
from qubosolver._io import utils as io_utils


def test_valid_qubo_passes_without_error() -> None:
    # A 5×5 QUBO with all coefficients >= 0 (identity matrix)
    coeffs = matrix.as_tensor(torch.eye(5))
    qi = Instance(coeffs)
    assert qi.size == 5
    # Verify that the tensor is stored correctly
    assert qi.matrix.shape == (5, 5)


def test_len_matches_size() -> None:
    qi = Instance(matrix.as_tensor(torch.eye(5)))
    check.equal(len(qi), 5)
    check.is_true(bool(qi))


def test_len_zero_is_falsy() -> None:
    qi = Instance(matrix.zeros(0))
    check.equal(len(qi), 0)
    check.is_false(bool(qi))


def test_size_exceeds_limit_triggers_system_exit() -> None:

    # An 81×81 QUBO exceeds the maximum supported size of 80×80
    coeffs = np.zeros((81, 81))

    qi = Instance(matrix.tensor(coeffs))
    # Expect SystemExit to be raised when setting oversized coefficients
    solver = Solver(qi, SolverConfig(solving=ClassicalSolvingConfig()))
    assert solver.instance.size == 81
    match_msg = "QUBO size 81×81 exceeds the maximum supported size of 80×80"
    with pytest.raises(ValueError, match=match_msg):
        Solver(qi, SolverConfig(solving=QuantumSolvingConfig()))


@pytest.mark.parametrize("size", [0, 1])
def test_max_off_diag_no_off_diag_entries(size: int) -> None:
    qi = Instance(matrix.zeros(size))
    with pytest.raises(RuntimeError, match="undefined"):
        qi._max_off_diag


def test_variable_fixing_property_raises_for_plain_instance() -> None:
    instance = Instance()
    check.is_not_instance(instance, transforms.variable_fixing.Instance)
    with pytest.raises(TypeError):
        instance.variable_fixing


def test_variable_fixing_property_returns_self_for_variable_fixing_instance() -> None:
    instance = transforms.variable_fixing.Instance(Instance())
    check.is_instance(instance, transforms.variable_fixing.Instance)
    check.is_(instance, instance.variable_fixing)


def test_zeroing_property_raises_for_plain_instance() -> None:
    instance = Instance()
    check.is_not_instance(instance, transforms.zeroing.Instance)
    with pytest.raises(TypeError):
        instance.zeroing


def test_zeroing_property_returns_self_for_zeroing_instance() -> None:
    instance = transforms.zeroing.Instance(Instance())
    check.is_instance(instance, transforms.zeroing.Instance)
    check.is_(instance, instance.zeroing)


def test_negative_bitflip_property_raises_for_plain_instance() -> None:
    instance = Instance()
    check.is_not_instance(instance, transforms.negative_bitflip.Instance)
    with pytest.raises(TypeError):
        instance.negative_bitflip


def test_negative_bitflip_property_returns_self_for_negative_bitflip_instance() -> None:
    instance = transforms.negative_bitflip.Instance(Instance())
    check.is_instance(instance, transforms.negative_bitflip.Instance)
    check.is_(instance, instance.negative_bitflip)


def test_save_load(simple_qubo_instance: Instance) -> None:

    file_path = Path(__file__).parent / "qubo_instance_test.pt"
    simple_qubo_instance.save(file_path)
    assert os.path.exists(file_path)
    loaded_instance = Instance.load(file_path)
    assert torch.allclose(loaded_instance.matrix, simple_qubo_instance.matrix)
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.mark.parametrize(
    "make_instance",
    [
        lambda parent: parent,
        lambda parent: transforms.variable_fixing.Instance(parent),
        lambda parent: transforms.zeroing.Instance(parent),
        lambda parent: transforms.negative_bitflip.Instance(parent),
    ],
    ids=["Instance", "variable_fixing", "zeroing", "negative_bitflip"],
)
def test_base_instance_load_dispatches_to_the_saved_type(make_instance, simple_qubo_instance: Instance) -> None:
    # The generic Instance.load() entry point must dispatch on the tag written
    # by save(), not on the class it's called through, so it must return
    # whichever concrete subclass was actually saved.
    instance = make_instance(simple_qubo_instance)

    buffer = io.BytesIO()
    instance.save(buffer)
    buffer.seek(0)
    loaded = Instance.load(buffer)

    check.equal(type(loaded), type(instance))
    torch.testing.assert_close(loaded.matrix, instance.matrix)


def test_subclass_load_rejects_a_stream_saved_as_a_different_type(simple_qubo_instance: Instance) -> None:
    # variable_fixing.Instance.load(f) must reject a stream that was saved as
    # a plain (or otherwise unrelated) Instance, instead of silently
    # returning the wrong type.
    buffer = io.BytesIO()
    simple_qubo_instance.save(buffer)
    buffer.seek(0)

    with pytest.raises(TypeError):
        transforms.variable_fixing.Instance.load(buffer)


def test_load_raises_on_unrecognized_tag() -> None:
    buffer = io.BytesIO()
    io_utils.save_string(buffer, "not.a.registered.Instance.subclass")
    buffer.seek(0)

    with pytest.raises(ValueError, match="unrecognized type tag"):
        Instance.load(buffer)
