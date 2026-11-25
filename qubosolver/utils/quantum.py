# Copyright (c) Pasqal, 2026
#
# Licensed under the Pasqal Open-Source Software License 2026 (MIT-derived).
# See LICENSE.md in the project root for the full terms.

"""Reconstruct QUBO instances from quantum register and drive parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qubosolver.types.instance import Instance

if TYPE_CHECKING:
    from collections.abc import Sequence

    import qoolqit
    from numpy.typing import NDArray


def _detuning(
    drive: qoolqit.Drive, t: float, *, n: int, dmm: bool = True, qubit_ids: Sequence[str | int] = []
) -> NDArray[np.float64]:

    delta = np.full(n, drive.detuning(t), dtype=np.float64)

    if dmm and drive.dmm is not None:
        if not qubit_ids:
            qubit_ids = list(drive.dmm.weights.keys())
        if set(qubit_ids) != set(drive.dmm.weights):
            msg = "drive.dmm.weights must have exactly the register's qubit ids."
            raise ValueError(msg)
        weights = np.asarray([drive.dmm.weights[i] for i in qubit_ids], dtype=np.float64)
        delta += drive.dmm.waveform(t) * weights

    return delta


def extract_qubo(register: qoolqit.Register, diagonal: qoolqit.Drive | Instance) -> Instance:
    """Reconstruct the QUBO encoded by a register's geometry and a diagonal.

    The off-diagonal coefficients come from the register's interactions
    (``1 / r ** 6``). The diagonal comes either from a drive's final detuning,
    so that the result is the QUBO the register and the drive implement, or
    from an instance's diagonal, e.g. the sub-instance a partial embedding was
    computed for, so that the result is the QUBO the register implements.

    Args:
        register: The physical register whose geometry encodes the QUBO's
            off-diagonal coefficients.
        diagonal: Either a drive, whose final detuning ``δ`` (and, if present,
            DMM) encodes the diagonal as ``-2 * δ``; or an instance whose
            diagonal is used as is, with variable *i* on qubit *i* of
            `register` (as returned by `instance.subinstance(variables)` for
            `variables` from a partial embedding).

    Returns:
        The reconstructed QUBO instance.

    Raises:
        ValueError: If `diagonal` is an instance whose size differs from the
            number of qubits in `register`.
    """
    Q = np.asarray(register.interaction_matrix(), dtype=np.float64)

    if isinstance(diagonal, Instance):
        if diagonal.size != len(register):
            msg = (
                f"The diagonal instance has {diagonal.size} variables but the register has "
                f"{len(register)} qubits."
            )
            raise ValueError(msg)
        Q += np.diag(diagonal.matrix.diagonal())
    else:
        delta = _detuning(
            diagonal, diagonal.duration, n=len(register), qubit_ids=register.qubits_ids
        )
        Q += np.diag(-2 * delta)

    return Instance(Q)


def _max_min_distance_ratio(device: qoolqit.Device) -> float:
    specs = device.specs
    min_distance = specs["min_distance"]
    max_radial_distance = specs["max_radial_distance"]
    if min_distance is not None and min_distance > 0 and max_radial_distance is not None:
        return max_radial_distance / min_distance
    return float("inf")
