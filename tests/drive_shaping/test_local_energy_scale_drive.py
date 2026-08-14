from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytest
import pytest_check as check
import torch

import qoolqit
from qoolqit import (
    AnalogDevice,
    DigitalAnalogDevice,
)

from qubosolver import (
    Analyzer,
    DriveShapingConfig,
    EmbeddingConfig,
    Instance,
    Solver,
    SolverConfig,
    matrix,
    tensor,
    vector,
)
from qubosolver.drive_shaping import (
    _local_energy_scale_drive,
)


@dataclass
class Solution:
    """Small helper representing a sampled solution."""

    bitstring: str
    cost: float = float("inf")
    probability: float = 0.0


def to_solutions(
    bitstrings: Iterable[str | torch.Tensor],
    costs: Iterable[float | torch.Tensor] = itertools.repeat(
        float("inf")
    ),
    probabilities: Iterable[
        float | torch.Tensor
    ] = itertools.repeat(0.0),
) -> list[Solution]:
    """Convert solver outputs into test solutions."""

    def to_string(
        bitstring: str | torch.Tensor,
    ) -> str:
        if isinstance(bitstring, torch.Tensor):
            return "".join(
                str(int(bit))
                for bit in bitstring
            )

        if isinstance(bitstring, str):
            return bitstring

        raise TypeError(
            "Unsupported bitstring type: "
            f"{type(bitstring).__name__}."
        )

    def to_float(
        value: float | torch.Tensor,
    ) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())

        return float(value)

    return [
        Solution(
            bitstring=to_string(bitstring),
            cost=to_float(cost),
            probability=to_float(probability),
        )
        for bitstring, cost, probability in zip(
            bitstrings,
            costs,
            probabilities,
        )
    ]

    
def gather_optimal_solutions(
    data: Iterable[Solution],
    min_cost: float | None = None,
) -> list[Solution]:
    """Return all solutions having the minimum cost."""
    data = list(data)

    if min_cost is None:
        min_cost = min(
            solution.cost
            for solution in data
        )

    return [
        solution
        for solution in data
        if np.allclose(
            solution.cost,
            min_cost,
        )
    ]


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "seed",
    [4548, 33671, 195530],
)
@pytest.mark.parametrize(
    "dmm",
    [True, False],
    ids=["dmm", "no_dmm"],
)
@pytest.mark.parametrize(
    "device_type",
    [
        DigitalAnalogDevice,
        AnalogDevice,
    ],
)
@pytest.mark.parametrize(
    "constant_diagonal",
    [True, False],
    ids=["cst_diag", "var_diag"],
)
@pytest.mark.parametrize(
    "diagonal_scale",
    [-0.9, -3.0, -1.5, -6.0],
)
def test_with_perfect_embedding(
    seed: int,
    dmm: bool,
    device_type: (
        type[DigitalAnalogDevice]
        | type[AnalogDevice]
    ),
    constant_diagonal: bool,
    diagonal_scale: float,
) -> None:
    """Test the heuristic on exactly embeddable QUBOs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def interaction_matrix_from_vertices(
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        interactions = (
            1.0
            / torch.cdist(
                vertices,
                vertices,
            )
            ** 6
        )

        interactions.fill_diagonal_(0.0)

        return interactions

    sqrt3 = np.sqrt(3.0)

    vertices = tensor.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.5, -0.5 * sqrt3],
            [-0.5, -0.5 * sqrt3],
        ]
    )

    diagonal = (
        torch.ones(
            4,
            dtype=vector.dtype(),
        )
        if constant_diagonal
        else vector.tensor(
            [
                1.0,
                1.25,
                0.2,
                1.167,
            ]
        )
    )

    qubo = (
        interaction_matrix_from_vertices(
            vertices
        )
        + diagonal_scale
        * torch.diag(diagonal)
    )

    qubo /= qubo.max()

    exact_solutions: list[Solution] = []

    for bits in itertools.product(
        [0, 1],
        repeat=4,
    ):
        bit_vector = tensor.tensor(bits)

        cost = float(
            (
                bit_vector
                @ qubo
                @ bit_vector
            ).item()
        )

        bitstring = "".join(
            str(int(bit))
            for bit in bit_vector.flatten()
        )

        exact_solutions.append(
            Solution(
                bitstring,
                cost,
            )
        )

    expected_optimal_solutions = (
        gather_optimal_solutions(
            exact_solutions
        )
    )

    check.is_not(
        expected_optimal_solutions,
        [],
    )

    print(
        "\nExpected minimum cost: "
        f"{expected_optimal_solutions[0].cost}"
    )


    expected_bitstrings = [
        solution.bitstring
        for solution in expected_optimal_solutions
    ]

    print(
        "Expected optimal bitstrings: "
        f"{expected_bitstrings}"
    )
    instance = Instance(
        matrix=qubo,
    )

    embedding_config = EmbeddingConfig(
        embedding_method="greedy",
        greedy_traps=100,
        greedy_max_possible_term=1,
    )

    drive_shaping_config = (
        DriveShapingConfig(
            drive_shaping_method=(
                "local_energy_scale"
            ),
            dmm=dmm,
            local_energy_scale_kappa=0.5,
        )
    )

    config = SolverConfig(
        use_quantum=True,
        embedding=embedding_config,
        drive_shaping=drive_shaping_config,
        device=device_type(),
    )

    solver = Solver(
        instance,
        config,
    )

    qubo_solution = solver.solve()
    qubo_solution.sort_by_cost()

    analyzer = Analyzer(
        [qubo_solution]
    )

    print(analyzer.df)

    register = solver.embedding()

    print(f"Register: {register.qubits}")
    print(
        f"Distances: {register.distances()}"
    )

    assert isinstance(
        qubo_solution.probabilities,
        torch.Tensor,
    )

    sampled_solutions = to_solutions(
        qubo_solution.bitstrings,
        qubo_solution.costs,
        qubo_solution.probabilities,
    )

    sampled_optimal_solutions = (
        gather_optimal_solutions(
            sampled_solutions
        )
    )

    check.is_not(
        sampled_optimal_solutions,
        [],
    )

    minimum_sampled_cost = (
        sampled_optimal_solutions[0].cost
    )

    print(
        "\nMinimum sampled cost: "
        f"{minimum_sampled_cost}"
    )
    sampled_best_bitstrings = [
        solution.bitstring
        for solution in sampled_optimal_solutions
    ]

    print(
        "Best sampled bitstrings: "
        f"{sampled_best_bitstrings}"
    )

    if (
        not constant_diagonal
        and not dmm
    ):
        pytest.skip(
            "DMM is required for variable "
            "diagonal coefficients."
        )

    if (
        not constant_diagonal
        and device_type is AnalogDevice
    ):
        pytest.skip(
            "AnalogDevice has no DMM and cannot "
            "encode variable diagonal coefficients."
        )

    expected_minimum_cost = (
        expected_optimal_solutions[0].cost
    )

    check.almost_equal(
        minimum_sampled_cost,
        expected_minimum_cost,
    )

    expected_bitstrings = [
        solution.bitstring
        for solution
        in expected_optimal_solutions
    ]

    for solution in sampled_optimal_solutions:
        check.is_in(
            solution.bitstring,
            expected_bitstrings,
        )

    cumulated_probability = sum(
        solution.probability
        for solution
        in sampled_optimal_solutions
    )

    check.greater(
        cumulated_probability,
        0.75,
    )


def test_too_high_diagonal() -> None:
    """Test amplitude clamping and detuning rescaling."""
    device = qoolqit.AnalogDeviceWithDMM()
    specs = device.specs

    eps = 0.001

    minimum_distance = specs["min_distance"]
    assert minimum_distance is not None
    minimum_distance += eps

    maximum_radius = specs[
        "max_radial_distance"
    ]
    assert maximum_radius is not None
    maximum_radius -= eps

    # Build a register that cannot be rescaled.
    register = (
        qoolqit.Register.from_coordinates(
            [
                [0.0, 0.0],
                [minimum_distance, 0.0],
                [maximum_radius, 0.0],
            ]
        )
    )

    qubo = (
        matrix.tensor(
            register.interaction_matrix()
        )
        + vector.zeros(3)
        .fill_(-50.0)
        .diag()
    )

    instance = Instance(qubo)

    with pytest.warns(
        UserWarning,
    ) as recorded_warnings:
        _local_energy_scale_drive.build_drive(
            instance,
            register,
            device=device,
        )

    warning_text = "\n".join(
        str(warning.message)
        for warning in recorded_warnings
    )

    max_amplitude = specs["max_amplitude"]
    max_detuning = specs["max_abs_detuning"]

    assert max_amplitude is not None
    assert max_detuning is not None

    amplitude_match = re.search(
        r"The local-energy-scale drive amplitude "
        r"\(([^)]+)\) exceeds the maximum amplitude "
        r"compilable on the device for this register "
        r"\(([^)]+)\); clamping to it\.",
        warning_text,
    )

    assert amplitude_match is not None

    check.almost_equal(
        float(amplitude_match.group(2)),
        max_amplitude,
        rel=1e-2,
    )

    detuning_match = re.search(
        r"The local-energy-scale detuning "
        r"\(([^)]+)\) exceeds the maximum detuning "
        r"compilable on the device for this amplitude "
        r"\(([^)]+)\); scaling the detuning down\.",
        warning_text,
    )

    assert detuning_match is not None

    check.almost_equal(
        float(detuning_match.group(2)),
        max_detuning,
        rel=1e-2,
    )