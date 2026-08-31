from __future__ import annotations

import numpy as np
import pytest
import torch
import pytest_check as check
from scipy.spatial.distance import pdist, squareform
from typing import Literal, get_args

import qoolqit
from qoolqit.graphs import DataGraph
from qubosolver import (
    Instance,
    Solution,
    Solver,
    matrix,
    QuantumSolvingConfig,
    DriveShapingConfig,
    SolverConfig,
    EmbeddingConfig,
)
from qubosolver.solver._drive_shaper import (
    _get_drive_shaper,
    BayesianSearchDriveShaper,
    ProportionalDiagonalDriveShaper,
)
from qubosolver.solver.config.drive_shaping import DriveShapingAlgorithm
from qubosolver.solver.solver import _QuboSolverQuantum


@pytest.fixture
def dummy_register() -> qoolqit.Register:
    qubits = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 3.0),
    }
    register = qoolqit.Register(qubits)
    return register


def test_generate_returns_drive_and_solution_proportional_diagonal(
    dummy_register: qoolqit.Register, simple_qubo_instance: Instance
) -> None:
    default_config = QuantumSolvingConfig()
    backend = default_config.backend
    shaper = _get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, qoolqit.Drive)
    assert isinstance(solution, Solution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities.numel() == 0
    assert solution.counts.numel() == 0


def test_generate_returns_drive_and_solution_bayesian_search(
    dummy_register: qoolqit.Register,
    simple_qubo_instance: Instance,
    bayesian_search_drive_shaping: DriveShapingConfig,
) -> None:
    default_config = QuantumSolvingConfig(drive_shaping=bayesian_search_drive_shaping)
    backend = default_config.backend
    shaper = ProportionalDiagonalDriveShaper(simple_qubo_instance, default_config, backend)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, qoolqit.Drive)
    assert isinstance(solution, Solution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert len(solution.probabilities) == 0
    assert len(solution.counts) == 0
    assert not solution


@pytest.mark.priority(35)
def test_generate_bayesian_search_drive_shaper(
    dummy_register: qoolqit.Register,
    simple_qubo_instance: Instance,
    bayesian_search_drive_shaping: DriveShapingConfig,
) -> None:
    default_config = QuantumSolvingConfig(
        drive_shaping=bayesian_search_drive_shaping,
        device=qoolqit.DigitalAnalogDevice(),
    )
    backend = default_config.backend
    shaper = _get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, BayesianSearchDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, qoolqit.Drive)
    assert isinstance(solution, Solution)
    assert solution.bitstrings.numel() > 0
    assert solution.costs.numel() > 0
    if isinstance(solution.probabilities, torch.Tensor):
        assert solution.probabilities.numel() > 0
    if isinstance(solution.counts, torch.Tensor):
        assert solution.counts.numel() > 0


@pytest.mark.priority(25)
@pytest.mark.parametrize("drive_method", get_args(DriveShapingAlgorithm))
@pytest.mark.parametrize("dmm", [True, False])
def test_normalized_weights_in_drive(
    drive_method: DriveShapingAlgorithm,
    dmm: bool,
    dummy_register: qoolqit.Register,
    simple_qubo_instance: Instance,
) -> None:
    # skip proportional-diagonal and local-energy-scale drive as their normalization is very specific.
    if dmm and drive_method in ["proportional_diagonal", "local_energy_scale"]:
        pytest.skip("Not implemented")
    default_config = QuantumSolvingConfig(
        drive_shaping=DriveShapingConfig(algorithm=drive_method, dmm=dmm),
    )
    backend = default_config.backend
    shaper = _get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    wdetuning = drive.dmm
    check.equal(dmm, wdetuning is not None)
    if wdetuning is None:
        return

    norm_weights = list(wdetuning.weights.values())
    weights = torch.abs(torch.diag(simple_qubo_instance.matrix)).tolist()
    max_w = max(weights)
    expected_norm = [(1 - (w / max_w)) for w in weights]

    assert pytest.approx(norm_weights, rel=1e-6) == expected_norm
    assert wdetuning.waveform.min() < 0


def test_drive_duration_set(
    dummy_register: qoolqit.Register, simple_qubo_instance: Instance
) -> None:
    default_config = QuantumSolvingConfig(device=qoolqit.DigitalAnalogDevice())
    backend = default_config.backend
    shaper = _get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    # DigitalAnalogDevice has a hardcoded duration
    check.almost_equal(drive.duration, 1000.0)


@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_generate_proportional_diagonal_drive_shaper(
    dummy_register: qoolqit.Register,
    simple_qubo_instance: Instance,
    dmm: bool,
) -> None:
    default_config = QuantumSolvingConfig(
        drive_shaping=DriveShapingConfig(
            algorithm="proportional_diagonal", dmm=dmm
        ),
        device=qoolqit.DigitalAnalogDevice(),
    )
    backend = default_config.backend
    shaper = _get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, ProportionalDiagonalDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, qoolqit.Drive)
    assert isinstance(solution, Solution)
    print("drive.duration =", drive.duration)

    check.almost_equal(drive.duration, 1000.0, abs=1.0e-3)
    check.almost_equal(drive.phase, 0.0)

    check.equal(drive.amplitude.duration, drive.duration)
    check.almost_equal(drive.amplitude.min(), 1e-9, abs=1e-10)
    check.almost_equal(drive.amplitude.max(), 0.375, abs=1e-4)

    check.equal(drive.detuning.duration, drive.duration)
    check.almost_equal(drive.detuning.min(), -1.5, abs=1e-4)
    if dmm:
        check.almost_equal(drive.detuning.max(), 1.5, abs=1e-4)
    else:
        check.almost_equal(drive.detuning.max(), 1.0, abs=1e-4)


@pytest.mark.usefixtures("restore_rng_state")
def test_shaper_does_not_overflow_device() -> None:
    data_graph = DataGraph.triangular(4, 4, 1)
    np.random.seed(0)
    removed = np.random.choice(
        data_graph.number_of_nodes(), data_graph.number_of_nodes() - 8, replace=False
    )
    data_graph.remove_nodes_from(removed)
    coords = [data_graph.coords[n] for n in data_graph.nodes]
    dist_matrix = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        qubo = 1.0 / dist_matrix**6
    np.fill_diagonal(qubo, -1.0)
    coefficients = np.asarray(qubo, dtype=float)

    config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm="greedy_layout"),
            drive_shaping=DriveShapingConfig(algorithm="proportional_diagonal"),
        )
    )
    solver = Solver(Instance(matrix.tensor(coefficients)), config)

    solution = solver.solve()
    assert solution.bitstrings.numel() > 0


def _embedding_drive_ratio(solver: Solver) -> float:
    """Return ``interaction(q0, q1) / final_detuning`` for a solved instance."""
    inner = solver._solver
    assert isinstance(inner, _QuboSolverQuantum)
    assert inner._cached_register is not None and inner._cached_drive is not None
    interactions = inner._cached_register.interactions()
    interaction = interactions[(0, 1)]
    detuning_wf = inner._cached_drive.detuning
    detuning = detuning_wf(detuning_wf.duration)
    return float(interaction / detuning)


@pytest.mark.parametrize("embedding_method", ["greedy_layout", "blade"])
def test_proportional_diagonal_register_and_drive_shape_normalization(
    embedding_method: Literal["greedy_layout", "blade"],
) -> None:
    """The proportional-diagonal drive shaper must normalize the pulse
    with the same convention as the register.
    """
    qubo = np.array([[-1.0, 2.0], [2.0, -1.0]])
    config = SolverConfig(
        solving=QuantumSolvingConfig(
            embedding=EmbeddingConfig(algorithm=embedding_method),
        )
    )
    solver = Solver(Instance(matrix.tensor(qubo)), config)
    solver.solve()

    ratio = _embedding_drive_ratio(solver)

    # Physically expected ratio: Q_ij encoded by the interaction, -0.5*Q_ii by
    # the final detuning -> 2.0 / (0.5) = 4.0
    expected = qubo[0, 1] / (-0.5 * qubo[0, 0])
    np.testing.assert_allclose(ratio, expected, rtol=1e-3)
