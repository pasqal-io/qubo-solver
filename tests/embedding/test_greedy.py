from __future__ import annotations


import torch
import numpy as np
from qoolqit.devices.device import BaseDevice
from qoolqit import AnalogDeviceWithDMM

from qubosolver.embedding._algorithms.greedy import Greedy
from qubosolver.embedding.greedy_layout import _resolve_max_possible_term
from qubosolver import Dataset, embedding, Instance, matrix
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
            [0.0, 1.0, 1.0 / 8.0, 1.0],
            [1.0, 0.0, 1.0, 1.0 / 8.0],
            [1.0 / 8.0, 1.0, 0.0, 1.0],
            [1.0, 1.0 / 8.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def assert_close_up_to_isometry(
    actual_vertices: torch.Tensor, expected_vertices: torch.Tensor, layout_angle: float
) -> None:
    A = torch.linalg.lstsq(expected_vertices, actual_vertices).solution
    # A has a symmetry. Apply a symmetry to get a pure rotation.
    if torch.linalg.det(A) < 0.0:
        A = torch.diag(torch.tensor([-1.0, 1.0])) @ A

    # A is a rotation matrix
    torch.testing.assert_close(A.T @ A, torch.eye(2))
    check.almost_equal(torch.linalg.det(A), 1.0)

    # Triangular Layout has a pi/3 rotation invariance
    # Square Layout has a pi/2 rotation invariance
    # The angle of rotation should be an integer multiple of it
    angle = torch.atan2(A[1, 0], A[0, 0]).item()
    normalized_angle = angle / layout_angle
    check.almost_equal(normalized_angle, round(normalized_angle), abs=1e-6)


def assert_close_to_lattice(vertices: torch.Tensor, basis: torch.Tensor) -> None:
    for v in vertices:
        v_lattice = basis.inverse() @ v
        torch.testing.assert_close(v_lattice, v_lattice.round())


def interaction_matrix_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
    n = vertices.shape[0]
    U = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            U[i, j] = 1.0 / torch.norm(vertices[i] - vertices[j]) ** 6
            U[j, i] = U[i, j]
    return U


@pytest.mark.parametrize("traps", [1, 2, 3, 6])
@pytest.mark.parametrize("relative_noise", [0.0, 0.01, 0.05, -0.01, -0.05])
def test_triangular_qubo(traps: int, relative_noise: float, max_min_dist_ratio: float) -> None:

    spacing = 7.0

    parameters = {
        "layout": embedding.Lattice.TRIANGULAR,
        "traps": traps,
        "spacing": spacing,
    }

    # Equilateral triangle
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5 * np.sqrt(3)],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    #  Matrix Q should match the spacing of the triangular layout so that the embedding returns
    # an equilateral triangle, hence the scale alpha.
    expected_U = interaction_matrix_from_vertices(expected_vertices)
    # All off-diagonal coefficients are equal to alpha
    alpha = expected_U[0, 1]
    Q = alpha * triangular_qubo() * (1.0 + relative_noise)
    # Tolerances from https://docs.pytorch.org/docs/stable/testing.html
    atol = 1e-5
    rtol = 1.3e-6 + abs(relative_noise)
    torch.testing.assert_close(Q, expected_U, atol=atol, rtol=rtol)

    if traps < 3:
        with pytest.raises(ValueError):
            Greedy().launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
        return

    result = Greedy().launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
    vertices = result[1]

    assert_close_up_to_isometry(vertices, expected_vertices, torch.pi / 3.0)
    # fmt: off
    basis = spacing * torch.tensor(
        [
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2.0],
        ],
        dtype=torch.float32,
    ).T
    # fmt: on
    assert_close_to_lattice(vertices, basis)

    U = interaction_matrix_from_vertices(vertices)
    torch.testing.assert_close(U, expected_U)


@pytest.mark.parametrize("traps", [1, 2, 6, 9])
@pytest.mark.parametrize("layout", [embedding.Lattice.SQUARE, "square"])
@pytest.mark.parametrize("relative_noise", [0.0, 0.01, 0.05, -0.01, -0.05])
def test_square_qubo(
    traps: int, layout: embedding.Lattice | str, relative_noise: float, max_min_dist_ratio: float
) -> None:

    spacing = 7.0

    parameters = {
        "layout": layout,
        "traps": traps,
        "spacing": spacing,
    }

    # Square
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    #  Matrix Q should match the spacing of the square layout so that the embedding returns
    # a square, hence the scale alpha.
    expected_U = interaction_matrix_from_vertices(expected_vertices)
    # All off-diagonal coefficients are equal to alpha
    alpha = expected_U[0, 1]
    Q = alpha * square_qubo() * (1.0 + relative_noise)
    # Tolerances from https://docs.pytorch.org/docs/stable/testing.html
    atol = 1e-5
    rtol = 1.3e-6 + abs(relative_noise)
    torch.testing.assert_close(Q, expected_U, atol=atol, rtol=rtol)

    if traps < 4:
        with pytest.raises(ValueError):
            Greedy().launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
        return

    result = Greedy().launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
    vertices = result[1]

    assert_close_up_to_isometry(vertices, expected_vertices, torch.pi / 2.0)
    # fmt: off
    basis = spacing * torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    ).T
    # fmt: on
    assert_close_to_lattice(vertices, basis)

    U = interaction_matrix_from_vertices(vertices)
    torch.testing.assert_close(U, expected_U)


@pytest.mark.parametrize("too_large", ["no", "barely", "extremely"])
@pytest.mark.parametrize("relative_noise", [0.0, 0.01, 0.05, -0.01, -0.05])
def test_too_large_spacing(
    too_large: str, relative_noise: float, device: BaseDevice, max_min_dist_ratio: float
) -> None:
    # A square layout of size 25 is composed of two concentric squares of side
    # 2*spacing and 4*spacing, plus the origin.
    layout = embedding.Lattice.SQUARE
    traps = 25

    assert isinstance(device.max_radial_distance, int)
    # Only the origin is within the device's maximum radial distance
    if too_large == "extremely":
        spacing = device.max_radial_distance * 3.0
        # max_min_dist_ratio = 1/3
    # Only the inner square is within the device's maximum radial distance
    elif too_large == "barely":
        # max_min_dist_ratio = np.sqrt(2)
        spacing = device.max_radial_distance / np.sqrt(2) - 0.1
    # All traps are within the device's maximum radial distance
    else:
        spacing = 7.0

    max_min_dist_ratio = max_min_dist_ratio * device.min_atom_distance / spacing

    parameters = {
        "layout": layout,
        "traps": traps,
        "spacing": spacing,
    }

    # Tailored QUBO to match the vertices below
    Q = torch.tensor(
        [
            [0.0, 1.0, 1.0 / 64.0],
            [1.0, 0.0, 1.0 / 125.0],
            [1.0 / 64.0, 1.0 / 125.0, 0.0],
        ],
        dtype=torch.float32,
    ) * (1.0 + relative_noise)

    # Tailored right triangle. With a correct spacing (e.g. 7.0):
    #   - Vertex 0 is at the origin
    #   - Vertex 1 is on the inner square
    #   - Vertex 2 is on the outer square
    expected_vertices = spacing * torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    #  Matrix Q should match the spacing of the square layout so that the embedding returns
    # an the expected right triangle, hence the scale alpha.
    expected_U = interaction_matrix_from_vertices(expected_vertices)
    # The spacing corresponds to the distance between vertices 0 and 1
    alpha = expected_U[0, 1]

    Q = alpha * Q
    # Tolerances from https://docs.pytorch.org/docs/stable/testing.html
    atol = 1e-5
    rtol = 1.3e-6 + abs(relative_noise)
    torch.testing.assert_close(Q, expected_U, atol=atol, rtol=rtol)

    greedy = Greedy()

    if too_large == "extremely":
        with pytest.raises(ValueError):
            greedy.launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
        return

    result = greedy.launch_greedy(Q=Q, params=parameters, max_min_dist_ratio=max_min_dist_ratio)
    vertices = result[1]
    U = interaction_matrix_from_vertices(vertices)

    # fmt: off
    basis = spacing * torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    ).T
    # fmt: on
    assert_close_to_lattice(vertices, basis)

    if too_large == "no":
        torch.testing.assert_close(U, expected_U)
        assert_close_up_to_isometry(vertices, expected_vertices, torch.pi / 2.0)

    if too_large == "barely":
        # expected_U is the interaction matrix if Q was perfectly embeddable
        # in the square layout, with the given parameters, which is not the case here
        expected_imperfect_U = expected_U
        expected_imperfect_U[0, 2] = expected_imperfect_U[2, 0] = 1.0 / 8.0 * expected_U[0, 1]
        torch.testing.assert_close(U, expected_imperfect_U)

        expected_imperfect_vertices = spacing * torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
        assert_close_up_to_isometry(vertices, expected_imperfect_vertices, torch.pi / 2.0)


def test_max_distance_constraint() -> None:

    device = AnalogDeviceWithDMM()._device

    # A square layout of size 9 is composed of a square of side 2*spacing, plus the origin.
    # With a large enough spacing, the outer corners should be out of range, and thus a
    # qubo of size 9 should not be embeddable.
    layout = embedding.Lattice.SQUARE
    traps = 9
    assert isinstance(device.max_radial_distance, int)
    spacing = 0.99 * device.max_radial_distance
    max_min_dist_ratio = device.max_radial_distance / spacing

    dataset = Dataset.from_random(1, traps)
    Q, _ = dataset[0]

    parameters = {
        "layout": layout,
        "traps": traps,
        "spacing": spacing,
    }

    with pytest.raises(ValueError):
        Greedy().launch_greedy(Q=Q.matrix, params=parameters, max_min_dist_ratio=max_min_dist_ratio)


def test_empty_embedding() -> None:
    config = embedding.greedy_layout.Config(traps=0)
    with pytest.raises(ValueError, match="empty instance"):
        embedding.greedy_layout.embed(Instance(), device=AnalogDeviceWithDMM(), config=config)


def test_single_atom_embedding() -> None:
    config = embedding.greedy_layout.Config(traps=1)
    instance = Instance(matrix.zeros(1))
    register = embedding.greedy_layout.embed(instance, device=AnalogDeviceWithDMM(), config=config)
    check.equal(len(register), 1)


def test_resolve_max_possible_term_float() -> None:
    instance = Instance(matrix.as_tensor(triangular_qubo()))
    check.equal(_resolve_max_possible_term(2.5, instance), 2.5)


def test_resolve_max_possible_term_factor() -> None:
    instance = Instance(matrix.as_tensor(triangular_qubo()))
    check.almost_equal(_resolve_max_possible_term(("factor", 2.0), instance), 2.0)


def test_resolve_max_possible_term_invalid_kind() -> None:
    instance = Instance(matrix.as_tensor(triangular_qubo()))
    with pytest.raises(ValueError, match="must be 'factor'"):
        _resolve_max_possible_term(("bogus", 2.0), instance)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [0, 1])
def test_resolve_max_possible_term_factor_no_off_diag_entries(size: int) -> None:
    instance = Instance(matrix.zeros(size))
    with pytest.raises(ValueError, match="fewer than"):
        _resolve_max_possible_term(("factor", 2.0), instance)
