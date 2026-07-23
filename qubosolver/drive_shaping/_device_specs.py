from __future__ import annotations

import math
import warnings

import qoolqit


def compare_specs(a: dict, b: dict) -> None:
    """Emit warnings for any key or value mismatches between two spec dicts.

    Args:
        a: First specification dictionary.
        b: Second specification dictionary.
    """
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    if keys_a != keys_b:
        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        if only_in_a:
            warnings.warn(f"Keys present in a but missing in b: {only_in_a}")
        if only_in_b:
            warnings.warn(f"Keys present in b but missing in a: {only_in_b}")

    for key in keys_a & keys_b:
        v1 = a[key]
        v2 = b[key]
        if v1 is None or v2 is None:
            continue
        if not math.isclose(float(v1), float(v2)):
            warnings.warn(f"Value mismatch for key '{key}': a={v1}, b={v2}")


def pulser_specs(
    device: qoolqit.Device, *, normalize: bool = False, check_against_qoolqit: bool = False
) -> dict[str, float | None]:
    """Extract hardware specifications from the underlying Pulser device.

    Args:
        device: A qoolqit device wrapping a Pulser device.
        normalize: If ``True``, normalize all values to natural units
            (distances by ``min_atom_distance``, energies by the nearest-neighbour
            interaction strength J₀).
        check_against_qoolqit: If ``True``, compare the result against
            `qoolqit_specs` and warn on mismatches.

    Returns:
        A dictionary of device specifications (max duration, amplitude, detuning, etc.).
    """
    pulser_device = device._device
    channel = pulser_device.channels["rydberg_global"]
    specs: dict[str, float | None] = {}
    specs["max_duration"] = (
        float(pulser_device.max_sequence_duration)
        if pulser_device.max_sequence_duration is not None
        else None
    )
    specs["max_amplitude"] = channel.max_amp
    specs["max_abs_detuning"] = channel.max_abs_detuning
    specs["min_distance"] = pulser_device.min_atom_distance
    specs["max_radial_distance"] = pulser_device.max_radial_distance
    specs["min_avg_amp"] = channel.min_avg_amp
    specs["dmm_bottom_detuning"] = None

    dmm_channels = list(getattr(pulser_device, "dmm_channels", {}).values())
    if dmm_channels:
        specs["dmm_bottom_detuning"] = getattr(dmm_channels[0], "bottom_detuning", None)

    if not normalize:
        return specs

    def _normalize(name: str, scale: float) -> None:
        if specs[name] is not None:
            specs[name] /= scale  # type: ignore[operator]

    r0 = specs["min_distance"]
    assert r0 is not None  # nosec B101
    _normalize("min_distance", r0)
    _normalize("max_radial_distance", r0)

    C6 = pulser_device.interaction_coeff
    J0 = C6 / (r0**6)

    _normalize("max_amplitude", J0)
    _normalize("max_abs_detuning", J0)
    _normalize("min_avg_amp", J0)
    _normalize("dmm_bottom_detuning", J0)

    # J0 is in rad/us and t in ns, hence the factor 1000
    _normalize("max_duration", 1000.0 / J0)

    if check_against_qoolqit:
        qq_specs = qoolqit_specs(device)
        compare_specs(specs, qq_specs)

    return specs


def qoolqit_specs(
    device: qoolqit.Device,
    *,
    complete_with_pulser: bool = False,
    check_against_pulser: bool = False,
) -> dict[str, float | None]:
    """Extract hardware specifications from the qoolqit device layer.

    Args:
        device: A qoolqit device.
        complete_with_pulser: If ``True``, fill missing keys from
            the underlying Pulser device specs.
        check_against_pulser: If ``True``, compare with `pulser_specs`
            and warn on mismatches.

    Returns:
        A dictionary of device specifications.
    """
    specs = device.specs
    _pulser_specs = pulser_specs(device, normalize=True)

    def import_from_pulser_or_set(name: str, fallback: float | None = None) -> None:
        if name in specs.keys():
            return
        if complete_with_pulser:
            specs[name] = _pulser_specs.get(name, fallback)
        else:
            specs[name] = fallback

    # Don't merge, update specific keys that are known not be in Qoolqit specs
    import_from_pulser_or_set("min_avg_amp")
    import_from_pulser_or_set("dmm_bottom_detuning")

    if check_against_pulser:
        compare_specs(specs, _pulser_specs)

    return specs
