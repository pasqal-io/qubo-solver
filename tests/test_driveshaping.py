from __future__ import annotations

import pytest
import torch
from pulser.devices import DigitalAnalogDevice, AnalogDevice
from qoolqit.register import Register
from qubosolver.config import DriveShapingConfig, SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.drive import (
    AdiabaticDriveShaper,
    OptimizedDriveShaper,
    get_drive_shaper,
)
from qubosolver.pipeline.targets import Pulse
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.qubo_types import DriveType


@pytest.fixture
def dummy_register() -> Register:
    register = Register.from_coordinates([(0.0, 0.0), (1.0, 0.0), (2.0, 3.0)])
    return register


def test_generate_returns_pulse_and_solution_adiabatic(
    dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = default_config.backend_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    pulse, solution = shaper.generate(dummy_register, simple_qubo_instance)

    assert isinstance(pulse, Pulse)
    assert isinstance(solution, QUBOSolution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities is None
    assert solution.counts is None


def test_generate_returns_pulse_and_solution_optimized(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
    optimized_pulse_shaping: DriveShapingConfig,
) -> None:
    default_config = SolverConfig(use_quantum=True, pulse_shaping=optimized_pulse_shaping)
    backend = default_config.backend_config.backend
    shaper = AdiabaticDriveShaper(simple_qubo_instance, default_config, backend)
    pulse, solution = shaper.generate(dummy_register, simple_qubo_instance)

    assert isinstance(pulse, Pulse)
    assert isinstance(solution, QUBOSolution)
    assert len(solution.bitstrings) == 0
    assert len(solution.costs) == 0
    assert solution.probabilities is None
    assert solution.counts is None


def test_generate_optimized_pulse_shaper(
    dummy_register: Register,
    simple_qubo_instance: QUBOInstance,
    optimized_pulse_shaping: DriveShapingConfig,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        pulse_shaping=optimized_pulse_shaping,
    )
    backend = default_config.backend_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, OptimizedDriveShaper)
    pulse, solution = shaper.generate(dummy_register, simple_qubo_instance)

    assert isinstance(pulse, Pulse)
    assert isinstance(solution, QUBOSolution)
    assert solution.bitstrings.numel() == 0  # empty tensor
    assert solution.costs.numel() == 0  # empty tensor
    if isinstance(solution.probabilities, torch.Tensor):
        assert solution.probabilities.numel() == 0
    if isinstance(solution.counts, torch.Tensor):
        assert solution.counts.numel() == 0  # empty tensor

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
    backend = default_config.backend_config.backend
    shaper = get_drive_shaper(
        simple_qubo_instance,
        SolverConfig(use_quantum=True, pulse_shaping=custom_fn_ps),
        backend,
    )
    assert isinstance(shaper, OptimizedDriveShaper)
    assert shaper.optimized_custom_objective_fn is not None
    assert shaper.optimized_callback_objective is not None
    assert shaper.optimized_custom_qubo_cost is not None
    pulse, solution = shaper.generate(dummy_register, simple_qubo_instance)
    assert len(opt_res) > 0
    assert opt_res[-1]["cost_eval"] == float(1e4)


@pytest.mark.parametrize("pulse_method", list(DriveType))
@pytest.mark.parametrize("dmm", [True, False])
def test_normalized_weights_in_pulse(
    pulse_method: str, dmm: bool, dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        pulse_shaping=DriveShapingConfig(drive_shaping_method=pulse_method, dmm=dmm),
    )
    backend = default_config.backend_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    pulse, _ = shaper.generate(dummy_register, simple_qubo_instance)

    norm_weights = pulse.norm_weights
    weights = torch.abs(torch.diag(simple_qubo_instance.coefficients)).tolist()
    max_w = max(weights)
    expected_norm = [1 - (w / max_w) for w in weights]

    assert pytest.approx(norm_weights, rel=1e-6) == expected_norm
    if dmm and pulse.final_detuning:
        assert pulse.final_detuning < 0


def test_pulse_duration_set(dummy_register: Register, simple_qubo_instance: QUBOInstance) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = default_config.backend_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, default_config, backend)
    pulse, _ = shaper.generate(dummy_register, simple_qubo_instance)

    # enforces AnalogDevice maximum sequence duration because Digital's one is a really specific number
    assert pulse.duration == AnalogDevice.max_sequence_duration


def test_custom_pulse_shaper(simple_qubo_instance: QUBOInstance) -> None:

    class MockAdiabaticPulseShaper(AdiabaticDriveShaper):
        pass

    config = SolverConfig(
        use_quantum=True,
        pulse_shaping=DriveShapingConfig(drive_shaping_method=MockAdiabaticPulseShaper),
    )
    backend = config.backend_config.backend
    shaper = get_drive_shaper(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockAdiabaticPulseShaper)
