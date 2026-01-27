from __future__ import annotations

import pytest
import torch
from pulser.devices import AnalogDevice
import pytest_check as check
from qoolqit.register import Register
from qoolqit.drive import Drive
from qubosolver.config import DriveShapingConfig, SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.drive import (
    AdiabaticDriveShaper,
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


def test_generate_returns_drive_and_solution_adiabatic(
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
    shaper = AdiabaticDriveShaper(simple_qubo_instance, default_config, backend)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities is None
    assert solution.counts is None


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


@pytest.mark.parametrize("drive_method", list(DriveType))
@pytest.mark.parametrize("dmm", [True, False])
def test_normalized_weights_in_drive(
    drive_method: str, dmm: bool, dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    if dmm and drive_method is DriveType.HEURISTIC:
        pytest.skip("Not implemented")

    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=drive_method, dmm=dmm),
    )
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    wdetuning = drive.weighted_detunings
    if len(wdetuning) > 0:
        assert len(wdetuning) == 1
        norm_weights = list(wdetuning[0].weights.values())
        weights = torch.abs(torch.diag(simple_qubo_instance.coefficients)).tolist()
        max_w = max(weights)
        TIME = default_config.device.converter.factors[0]
        expected_norm = [(1 - (w / max_w)) / TIME for w in weights]

        assert pytest.approx(norm_weights, rel=1e-6) == expected_norm
        assert wdetuning[0].waveform.min() < 0


def test_drive_duration_set(dummy_register: Register, simple_qubo_instance: QUBOInstance) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    drive, _ = shaper.generate(dummy_register)

    # enforces AnalogDevice maximum sequence duration because Digital's one is a really specific number
    assert (
        drive.duration * default_config.device.converter.factors[0]
        == AnalogDevice.max_sequence_duration
    )


def test_custom_drive_shaper(simple_qubo_instance: QUBOInstance) -> None:

    class MockAdiabaticdriveShaper(AdiabaticDriveShaper):
        pass

    config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=MockAdiabaticdriveShaper),
    )
    backend = config.backend
    shaper = get_drive_shaper(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockAdiabaticdriveShaper)


@pytest.mark.parametrize("dmm", [True, False])
def test_heuristic_drive_get_hw_dmm_bound(dmm: bool, simple_qubo_instance: QUBOInstance) -> None:

    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=DriveType.HEURISTIC, dmm=dmm),
    )
    shaper = get_drive_shaper(simple_qubo_instance, default_config, default_config.backend)
    assert isinstance(shaper, HeuristicDriveShaper)

    check.is_none(shaper._get_hw_dmm_bound())


def test_heuristic_drive_compute_alpha_diag_max(simple_qubo_instance: QUBOInstance) -> None:

    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=DriveType.HEURISTIC),
    )
    shaper = get_drive_shaper(simple_qubo_instance, default_config, default_config.backend)
    assert isinstance(shaper, HeuristicDriveShaper)

    check.almost_equal(shaper._compute_alpha_diag_max(0.0, 0.0, -2.0, 5.0, 3.0), 1.0)
    check.almost_equal(shaper._compute_alpha_diag_max(-2.0, -2.0, -2.0, 5.0, 3.0), 2.5)
    check.almost_equal(shaper._compute_alpha_diag_max(2.0, 2.0, -2.0, 5.0, 3.0), 1.0)

    check.almost_equal(shaper._compute_alpha_diag_max(0.0, 4.0, -2.0, 5.0, 3.0), 0.75)
    check.almost_equal(shaper._compute_alpha_diag_max(-2.0, 4.0, -2.0, 5.0, 3.0), 0.5)
    check.almost_equal(shaper._compute_alpha_diag_max(2.0, 4.0, -2.0, 5.0, 3.0), 1.0)


def test_generate_heuristic_drive_shaper(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=DriveType.HEURISTIC),
    )
    backend = default_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, HeuristicDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    check.equal(solution.solution_status, SolutionStatusType.UNPROCESSED)

    check.almost_equal(drive.duration, 94.248, abs=1.0e-3)
    check.almost_equal(drive.phase, 0.0)

    check.equal(drive.amplitude.duration, drive.duration)
    check.almost_equal(drive.amplitude.min(), 0.0, abs=1e-10)
    check.almost_equal(drive.amplitude.max(), 0.3948, abs=1e-4)
    check.equal(drive.amplitude._interpolator, "PchipInterpolator")

    check.equal(drive.detuning.duration, drive.duration)
    check.almost_equal(drive.detuning.min(), -1.9739, abs=1e-4)
    check.almost_equal(drive.detuning.max(), 1.5791, abs=1e-4)
    check.equal(drive.detuning._interpolator, "PchipInterpolator")


def test_generate_trivial_heuristic_drive_shaper(
    dummy_register: Register,
    trivial_qubo_instance: QUBOInstance,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        drive_shaping=DriveShapingConfig(drive_shaping_method=DriveType.HEURISTIC),
    )
    backend = default_config.backend
    shaper = get_drive_shaper(trivial_qubo_instance, default_config, backend)
    assert isinstance(shaper, HeuristicDriveShaper)
    drive, solution = shaper.generate(dummy_register)

    assert isinstance(drive, Drive)
    assert isinstance(solution, QUBOSolution)
    check.equal(solution.solution_status, SolutionStatusType.UNPROCESSED)

    check.almost_equal(drive.duration, 94.248, abs=1.0e-3)
    check.almost_equal(drive.phase, 0.0)

    check.equal(drive.amplitude.duration, drive.duration)
    check.almost_equal(drive.amplitude.min(), 0.0, abs=1e-10)
    check.almost_equal(drive.amplitude.max(), 0.0, abs=1e-10)
    check.equal(drive.amplitude._interpolator, "PchipInterpolator")

    check.equal(drive.detuning.duration, drive.duration)
    check.almost_equal(drive.detuning.min(), 0.0, abs=1e-10)
    check.almost_equal(drive.detuning.max(), 0.0, abs=1e-10)
    check.equal(drive.detuning._interpolator, "PchipInterpolator")
