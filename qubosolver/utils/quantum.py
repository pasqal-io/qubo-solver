from __future__ import annotations

import torch

import qoolqit
from qubosolver.types.instance import Instance
from qubosolver.types.linalg import Vector
from qubosolver.types import vector, matrix
from typing import Sequence


def _detuning(
    drive: qoolqit.Drive, t: float, *, n: int, dmm: bool = True, qubit_ids: Sequence[str | int] = []
) -> Vector:

    delta = vector.zeros(n)
    delta.fill_(drive.detuning(t))

    if dmm and drive.dmm is not None:
        if not qubit_ids:
            qubit_ids = list(drive.dmm.weights.keys())
        if set(qubit_ids) != set(drive.dmm.weights):
            raise ValueError("drive.dmm.weights must have exactly the register's qubit ids.")
        weights = vector.tensor([drive.dmm.weights[i] for i in qubit_ids])
        delta += drive.dmm.waveform(t) * weights

    return delta


def extract_qubo(register: qoolqit.Register, drive: qoolqit.Drive) -> Instance:
    """Reconstruct the QUBO encoded by a register's geometry and a drive's final detuning.

    Off-diagonal coefficients are read from the register's pairwise ``1/r^6``
    interaction strengths. Diagonal coefficients are recovered from the
    drive's final detuning value(s), inverting the
    ``d_i = -0.5 * Q[i, i]`` convention used when shaping a drive from a
    QUBO instance (see `qubosolver.drive_shaping.proportional_diagonal.build_drive`).

    Args:
        register: The physical register whose geometry encodes the QUBO's
            off-diagonal coefficients.
        drive: The drive whose final detuning (and, if present, DMM)
            encodes the QUBO's diagonal coefficients.

    Returns:
        The reconstructed QUBO instance.
    """
    Q = matrix.tensor(register.interaction_matrix())

    delta = _detuning(drive, drive.duration, n=len(register), qubit_ids=register.qubits_ids)
    Q += torch.diag(-2 * delta)

    return Instance(Q)
