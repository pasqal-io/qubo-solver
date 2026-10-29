# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

from __future__ import annotations

import logging

import pytest
import pytest_check as check
import qoolqit

from qubosolver.drive_shaping._device_specs import (
    compare_specs,
    max_virtual_amplitude,
    pulser_specs,
    qoolqit_specs,
)


@pytest.mark.parametrize("check_against_qoolqit", [False, True])
def test_pulser_specs_normalized(
    check_against_qoolqit: bool, caplog: pytest.LogCaptureFixture
) -> None:
    device = qoolqit.DigitalAnalogDevice()
    with caplog.at_level(logging.WARNING):
        specs = pulser_specs(device, normalize=True, check_against_qoolqit=check_against_qoolqit)
    check.equal(caplog.records, [])
    check.equal(specs["min_distance"], 1.0)
    check.is_not_none(specs["dmm_bottom_detuning"])
    for key, value in device.specs.items():
        check.almost_equal(specs[key], value)


@pytest.mark.parametrize("complete_with_pulser", [False, True])
def test_qoolqit_specs_completion(complete_with_pulser: bool) -> None:
    device = qoolqit.DigitalAnalogDevice()
    specs = qoolqit_specs(device, complete_with_pulser=complete_with_pulser)
    expected = pulser_specs(device, normalize=True) if complete_with_pulser else {}
    check.equal(specs["min_avg_amp"], expected.get("min_avg_amp"))
    check.equal(specs["dmm_bottom_detuning"], expected.get("dmm_bottom_detuning"))


def test_qoolqit_specs_keeps_existing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    device = qoolqit.DigitalAnalogDevice()
    device_specs = {**device.specs, "min_avg_amp": 0.5}
    monkeypatch.setattr(type(device), "specs", property(lambda _: dict(device_specs)))
    specs = qoolqit_specs(device, complete_with_pulser=True)
    check.equal(specs["min_avg_amp"], 0.5)


def test_qoolqit_specs_check_against_pulser(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        qoolqit_specs(qoolqit.DigitalAnalogDevice(), check_against_pulser=True)
    check.equal(caplog.records, [])


def test_compare_specs_warns_on_missing_keys(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        compare_specs({"x": 1.0, "a": 1.0}, {"x": 1.0, "b": 1.0})
    messages = caplog.messages
    check.equal(len(messages), 2)
    check.is_in("present in a but missing in b: {'a'}", messages[0])
    check.is_in("present in b but missing in a: {'b'}", messages[1])


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ({"x": 1.0, "a": 1.0}, {"x": 1.0}, "Keys present in a but missing in b: {'a'}"),
        ({"x": 1.0}, {"x": 1.0, "b": 1.0}, "Keys present in b but missing in a: {'b'}"),
    ],
)
def test_compare_specs_warns_on_one_sided_missing_keys(
    a: dict, b: dict, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        compare_specs(a, b)
    check.equal(caplog.messages, [message])


def test_compare_specs_warns_on_value_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        compare_specs({"x": 1.0, "y": None}, {"x": 2.0, "y": 3.0})
    check.equal(caplog.messages, ["Value mismatch for key 'x': a=1.0, b=2.0"])


def test_compare_specs_ignores_none_and_matching_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        compare_specs({"x": 1.0, "y": None}, {"x": 1.0, "y": 3.0})
    check.equal(caplog.records, [])


def test_max_virtual_amplitude_zero_radial_extent() -> None:
    device = qoolqit.AnalogDevice()
    register = qoolqit.Register({"q0": (0.0, 0.0), "q1": (0.0, 0.0)})
    check.equal(max_virtual_amplitude(device, register), device.specs["max_amplitude"])
