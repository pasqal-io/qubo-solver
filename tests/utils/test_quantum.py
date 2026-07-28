from __future__ import annotations

import math

import pytest_check as check
import torch

import qoolqit
from qoolqit import Constant as ConstantWaveform
from qoolqit import Ramp as RampWaveform
from qubosolver import Instance, matrix, vector, drive_shaping, embedding, extract_qubo, LayoutType


def _register() -> qoolqit.Register:
    return qoolqit.Register(
        {
            "0": (0.0, 0.0),
            "1": (1.0, 0.0),
            "2": (0.0, 2.0),
        }
    )


def test_extract_qubo_off_diagonal_matches_interactions() -> None:
    register = _register()
    # No detuning
    drive = qoolqit.Drive(amplitude=ConstantWaveform(10.0, 1.0))

    instance = extract_qubo(register, drive)

    r01, r02, r12 = 1.0, 2.0, math.sqrt(5)
    expected = matrix.tensor(
        [
            [0.0, 1.0 / r01**6, 1.0 / r02**6],
            [1.0 / r01**6, 0.0, 1.0 / r12**6],
            [1.0 / r02**6, 1.0 / r12**6, 0.0],
        ]
    )

    check.equal(instance.size, 3)
    torch.testing.assert_close(instance.matrix, expected)


def test_extract_qubo_diagonal_from_constant_detuning() -> None:
    register = _register()
    drive = qoolqit.Drive(
        amplitude=ConstantWaveform(10.0, 1.0),
        detuning=ConstantWaveform(10.0, -3.0),
    )

    instance = extract_qubo(register, drive)

    expected_diagonal = vector.tensor([6.0, 6.0, 6.0])  # -2 * (-3.0)
    torch.testing.assert_close(torch.diag(instance.matrix), expected_diagonal)


def test_extract_qubo_diagonal_from_ramp_detuning_uses_final_value() -> None:
    register = _register()
    drive = qoolqit.Drive(
        amplitude=ConstantWaveform(10.0, 1.0),
        detuning=RampWaveform(10.0, 0.0, -2.0),
    )

    instance = extract_qubo(register, drive)

    expected_diagonal = vector.tensor([4.0, 4.0, 4.0])  # -2 * -2.0
    torch.testing.assert_close(torch.diag(instance.matrix), expected_diagonal)


def test_extract_qubo_diagonal_with_dmm_weights_per_atom() -> None:
    register = _register()
    dmm = qoolqit.drive.DetuningMapModulator(
        waveform=ConstantWaveform(10.0, -1.0),
        weights={"0": 1.0, "1": 0.5, "2": 0.0},
    )
    drive = qoolqit.Drive(
        amplitude=ConstantWaveform(10.0, 1.0),
        detuning=ConstantWaveform(10.0, -2.0),
        dmm=dmm,
    )

    instance = extract_qubo(register, drive)

    # Q[i,i] = -2 * (delta_g_T + dmm_final * weight_i)
    expected_diagonal = vector.tensor(
        [
            -2 * (-2.0 + -1.0 * 1.0),
            -2 * (-2.0 + -1.0 * 0.5),
            -2 * (-2.0 + -1.0 * 0.0),
        ]
    )
    torch.testing.assert_close(torch.diag(instance.matrix), expected_diagonal)


def test_extract_qubo_matrix_is_symmetric() -> None:
    register = _register()
    drive = qoolqit.Drive(
        amplitude=ConstantWaveform(10.0, 1.0),
        detuning=ConstantWaveform(10.0, -1.0),
    )

    instance = extract_qubo(register, drive)

    torch.testing.assert_close(instance.matrix, instance.matrix.T)


def test_extract_qubo_round_trip_through_greedy_embedding_and_drive_shaping() -> None:
    # A register of 3 atoms sitting on qoolqit's triangular lattice layout
    # (the grid greedy embedding places atoms on), picked non-adjacent so
    # the resulting triangle is scalene rather than equilateral.
    lattice = LayoutType.TRIANGULAR.value(n_traps=12, spacing=1.0)
    coords = lattice.coords
    triangle = qoolqit.Register(
        {
            "0": tuple(coords[5]),
            "1": tuple(coords[1]),
            "2": tuple(coords[10]),
        }
    )
    index = {"0": 0, "1": 1, "2": 2}
    Q = matrix.zeros(3)
    for (u, v), value in triangle.interactions().items():
        Q[index[u], index[v]] = value
        Q[index[v], index[u]] = value
    Q.diagonal().copy_(vector.tensor([-1.0, -0.5, -1.1]))

    original = Instance(matrix=Q)

    device = qoolqit.AnalogDeviceWithDMM()
    config = embedding.greedy.Config(traps=12, max_possible_term=1.0)
    register = embedding.greedy.embed(original, device, config=config)
    drive = drive_shaping.heuristic.build_drive(original, register, device=device, dmm=True)

    extracted = extract_qubo(register, drive)
    torch.testing.assert_close(extracted.matrix, original.matrix)

    # Example metrics to assess the quality of the embedding and drive
    # shaping: the Frobenius distance between the original and extracted
    # QUBO, on the full matrix and on the diagonal/off-diagonal parts
    # separately (which are encoded independently, by the register's
    # geometry and the drive's detuning, respectively).
    full_distance = torch.linalg.norm(extracted.matrix - original.matrix).item()
    off_diag_mask = ~torch.eye(original.size, dtype=torch.bool)
    off_diag_distance = torch.linalg.norm(
        extracted.matrix[off_diag_mask] - original.matrix[off_diag_mask]
    ).item()
    diag_distance = torch.linalg.norm(
        torch.diag(extracted.matrix) - torch.diag(original.matrix)
    ).item()
    check.almost_equal(full_distance, 0.0, abs=1e-6)
    check.almost_equal(off_diag_distance, 0.0, abs=1e-6)
    check.almost_equal(diag_distance, 0.0, abs=1e-6)
