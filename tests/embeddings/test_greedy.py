# tests/test_greedy_annimation.py

from __future__ import annotations


import torch
import numpy as np
from qoolqit.devices.device import DigitalAnalogDevice

from qubosolver.algorithms.greedy.greedy import Greedy
from qubosolver.qubo_types import LayoutType
import pytest
import pytest_check as check


def triangular_qubo() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def square_qubo() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 1.0/8.0, 1.0],
            [1.0, 0.0, 1.0, 1.0/8.0],
            [1.0/8.0, 1.0, 0.0, 1.0],
            [1.0, 1.0/8.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def interaction_matrix_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
    n = vertices.shape[0]
    U = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            U[i, j] = 1.0 / torch.norm(vertices[i] - vertices[j]) ** 6
            U[j, i] = U[i, j]
    return U


@pytest.mark.parametrize("traps", [3, 6])
def test_triangular_qubo(traps: int) -> None:

    spacing = 7.0

    device = DigitalAnalogDevice()._device
    C6 = device.interaction_coeff

    parameters = {
        "layout": LayoutType.TRIANGULAR,
        "traps": traps,
        "spacing": spacing,
        "device": device,
    }

    # Equilateral triangle
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5 * np.sqrt(3)],
            [1.0, 0.0],
        ]
    )
    #  Matrix Q should match the spacing of the triangular layout so that the embedding returns
    # an equilateral triangle, hence the scale alpha.
    expected_U = C6 * interaction_matrix_from_vertices(expected_vertices)
    # All off-diagonal coefficients are equal to alpha
    alpha = expected_U[0, 1]
    Q = alpha * triangular_qubo()
    torch.testing.assert_close(Q, expected_U)

    result = Greedy().launch_greedy(Q=Q, params=parameters)
    vertices = result[0][1]["coords"]
    # Vertices form an equilateral triangle of side = spacing
    for i in range(3):
        for j in range(i + 1, 3):
            d = torch.dist(vertices[i, :], vertices[j, :])
            check.almost_equal(d, spacing)

    U = C6 * interaction_matrix_from_vertices(vertices)
    torch.testing.assert_close(U, expected_U)

@pytest.mark.parametrize("traps", [4, 9])
@pytest.mark.parametrize("layout", [LayoutType.SQUARE, "square"])
def test_square_qubo(traps: int, layout: LayoutType | str) -> None:

    spacing = 7.0

    device = DigitalAnalogDevice()._device
    C6 = device.interaction_coeff

    parameters = {
        "layout": layout,
        "traps": traps,
        "spacing": spacing,
        "device": device,
    }

    # Equilateral triangle
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    #  Matrix Q should match the spacing of the square layout so that the embedding returns
    # a square, hence the scale alpha.
    expected_U = C6 * interaction_matrix_from_vertices(expected_vertices)
    # All off-diagonal coefficients are equal to alpha
    alpha = expected_U[0, 1]
    Q = alpha * square_qubo()
    torch.testing.assert_close(Q, expected_U)

    result = Greedy().launch_greedy(Q=Q, params=parameters)
    vertices = result[0][1]["coords"]

    def sorted(vertices: torch.Tensor) -> torch.Tensor:
        indices = np.lexsort(vertices.abs().numpy().T)
        return vertices[indices,:].abs()

    # Vertices form square of side = spacing
    torch.testing.assert_close(sorted(vertices), sorted(expected_vertices))

    U = C6 * interaction_matrix_from_vertices(vertices)
    torch.testing.assert_close(U, expected_U)

@pytest.mark.xfail(strict=False, reason="bug not fixed yet")
def test_triangular_qubo_not_enough_traps() -> None:

    spacing = 7.0

    device = DigitalAnalogDevice()._device
    C6 = device.interaction_coeff

    parameters = {
        "layout": LayoutType.TRIANGULAR,
        "traps": 2,
        "spacing": spacing,
        "device": device,
    }

    # Equilateral triangle
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5 * np.sqrt(3)],
            [1.0, 0.0],
        ]
    )
    #  Matrix Q should match the spacing of the triangular layout so that the embedding returns
    # an equilateral triangle, hence the scale alpha.
    expected_U = C6 * interaction_matrix_from_vertices(expected_vertices)
    # All off-diagonal coefficients are equal to alpha
    alpha = expected_U[0, 1]
    Q = alpha * triangular_qubo()
    torch.testing.assert_close(Q, expected_U)

    result = Greedy().launch_greedy(Q=Q, params=parameters)
    vertices = result[0][1]["coords"]
    # Vertices form an equilateral triangle of side = spacing
    for i in range(3):
        for j in range(i + 1, 3):
            d = torch.dist(vertices[i, :], vertices[j, :])
            check.almost_equal(d, spacing)

    U = C6 * interaction_matrix_from_vertices(vertices)
    torch.testing.assert_close(U, expected_U)
