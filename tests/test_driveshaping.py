from __future__ import annotations

import pytest
import torch
import pytest_check as check
from qoolqit.register import Register
from qoolqit.drive import Drive
from qubosolver.config import DriveShapingConfig, SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.drive import (
    OptimizedDriveShaper,
    HeuristicDriveShaper,
    get_drive_shaper,
)
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.qubo_types import DriveType, SolutionStatusType


@pytest.fixture
def dummy_register() -> Register:
    register = Register.from_coordinates([(0.0, 0.0), (1.0, 0.0), (2.0, 3.0)])
    return register


def test_generate_returns_drive_and_solution_heuristic(
    dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities is None
    assert solution.counts is None


def test_generate_returns_drive_and_solution_optimized(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
    optimized_drive_shaping: DriveShapingConfig,
) -> None:
    default_config = SolverConfig(use_quantum=True, drive_shaping=optimized_drive_shaping)
    backend = default_config.backend
    shaper = HeuristicDriveShaper(simple_qubo_instance, default_config, backend)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities is None
    assert solution.counts is None


@pytest.mark.priority(35)
def test_generate_optimized_drive_shaper(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
    optimized_drive_shaping: DriveShapingConfig,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=optimized_drive_shaping,
    )
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, OptimizedDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    assert solution.bitstrings.numel() > 0
    assert solution.costs.numel() > 0
    if isinstance(solution.probabilities, torch.Tensor):
        assert solution.probabilities.numel() > 0
    if isinstance(solution.counts, torch.Tensor):
        assert solution.counts.numel() > 0

    # try with custom objective_fn

    def custom_ojective(
        bitstrings: list,
        counts: list,
        probabilities: list,
        costs: list,
        best_cost: float,
        best_bitstring: str,
    ) -> float:
        return float(1e4)

    opt_res = []

    def callback_fn(d: dict) -> None:
        opt_res.append(d)

    def custom_qubo(bitstring: str, QUBO: torch.Tensor) -> float:
        return 1.0

    custom_fn_ps = DriveShapingConfig(
        drive_shaping_method=default_config.drive_shaping.drive_shaping_method,
        optimized_custom_objective=custom_ojective,
        optimized_callback_objective=callback_fn,
        optimized_custom_qubo_cost=custom_qubo,
    )
    backend = default_config.backend
    shaper = get_drive_shaper(
        simple_qubo_instance,
        SolverConfig(use_quantum=True, drive_shaping=custom_fn_ps),
        backend,
    )
    assert isinstance(shaper, OptimizedDriveShaper)
    assert shaper.optimized_custom_objective_fn is not None
    assert shaper.optimized_callback_objective is not None
    assert shaper.optimized_custom_qubo_cost is not None
    drive, solution = shaper.generate(dummy_register)
    assert len(opt_res) > 0
    assert opt_res[-1]["cost_eval"] == float(1e4)


@pytest.mark.priority(25)
@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("dmm", [True, False])
def test_normalized_weights_in_drive(
    drive_method: DriveType, dmm: bool, dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    # skip heuristic-drive as its normalization is very specific
    if dmm and drive_method is DriveType.HEURISTIC:
        pytest.skip("Not implemented")
    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=drive_method, dmm=dmm),
    )
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    wdetuning = drive.dmm
    check.equal(dmm, wdetuning is not None)
    if wdetuning is None:
        return

    norm_weights = list(wdetuning.weights.values())
    weights = torch.abs(torch.diag(simple_qubo_instance.coefficients)).tolist()
    max_w = max(weights)
    expected_norm = [(1 - (w / max_w)) for w in weights]

    assert pytest.approx(norm_weights, rel=1e-6) == expected_norm
    assert wdetuning.waveform.min() < 0


def test_drive_duration_set(dummy_register: Register, simple_qubo_instance: QUBOInstance) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    # DigitalAnalogDevice has a hardcoded duration
    check.almost_equal(drive.duration, 1000.0)


def test_custom_drive_shaper(simple_qubo_instance: QUBOInstance) -> None:

    class MockHeuristicDriveShaper(HeuristicDriveShaper):
        pass

    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=MockHeuristicDriveShaper),
    )
    backend = config.backend
    shaper = get_drive_shaper(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockHeuristicDriveShaper)


@pytest.mark.parametrize("dmm", [True, False], ids=["dmm", "no_dmm"])
def test_generate_heuristic_drive_shaper(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
    dmm: bool,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=DriveType.HEURISTIC, dmm=dmm),
    )
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, HeuristicDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    print("drive.duration =", drive.duration)

    check.equal(solution.solution_status, SolutionStatusType.UNPROCESSED)

    check.almost_equal(drive.duration, 1000.0, abs=1.0e-3)
    check.almost_equal(drive.phase, 0.0)

    check.equal(drive.amplitude.duration, drive.duration)
    check.almost_equal(drive.amplitude.min(), 1e-9, abs=1e-10)
    check.almost_equal(drive.amplitude.max(), 0.75, abs=1e-4)

    check.equal(drive.detuning.duration, drive.duration)
    check.almost_equal(drive.detuning.min(), -3.0, abs=1e-4)
    if dmm:
        check.almost_equal(drive.detuning.max(), 3.0, abs=1e-4)
    else:
        check.almost_equal(drive.detuning.max(), 2.0, abs=1e-4)
