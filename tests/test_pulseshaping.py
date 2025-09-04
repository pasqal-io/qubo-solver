from __future__ import annotations

import pytest
import torch
from pulser.devices import DigitalAnalogDevice
from qoolqit._solvers import get_backend

from qubosolver.config import PulseShapingConfig, SolverConfig
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.pulse import (
    AdiabaticPulseShaper,
    OptimizedPulseShaper,
    get_pulse_shaper,
)
from qubosolver.pipeline.targets import Pulse, Register
from qubosolver.qubo_instance import QUBOInstance


@pytest.fixture
def dummy_register() -> Register:
    register = Register(
        device=DigitalAnalogDevice, register=[(0.0, 0.0), (1.0, 0.0), (2.0, 3.0)]
    )
    return register


def test_generate_returns_pulse_and_solution_adiabatic(
    dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = get_backend(default_config.backend_config)
    shaper = get_pulse_shaper(simple_qubo_instance, default_config, backend)
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
    optimized_pulse_shaping: PulseShapingConfig,
) -> None:
    default_config = SolverConfig(
        use_quantum=True, pulse_shaping=optimized_pulse_shaping
    )
    backend = get_backend(default_config.backend_config)
    shaper = AdiabaticPulseShaper(simple_qubo_instance, default_config, backend)
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
    optimized_pulse_shaping: PulseShapingConfig,
) -> None:
    default_config = SolverConfig(
        use_quantum=True,
        pulse_shaping=optimized_pulse_shaping,
    )
    backend = get_backend(default_config.backend_config)
    shaper = get_pulse_shaper(simple_qubo_instance, default_config, backend)
    assert isinstance(shaper, OptimizedPulseShaper)
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

    custom_fn_ps = PulseShapingConfig(
        pulse_shaping_method=default_config.pulse_shaping.pulse_shaping_method,
        custom_objective=custom_ojective,
        callback_objective=callback_fn,
        custom_qubo_cost=custom_qubo,
    )
    backend = get_backend(default_config.backend_config)
    shaper = get_pulse_shaper(
        simple_qubo_instance,
        SolverConfig(use_quantum=True, pulse_shaping=custom_fn_ps),
        backend,
    )
    assert isinstance(shaper, OptimizedPulseShaper)
    assert shaper.custom_objective_fn is not None
    assert shaper.callback_objective is not None
    assert shaper.custom_qubo_cost is not None
    pulse, solution = shaper.generate(dummy_register, simple_qubo_instance)
    assert len(opt_res) > 0
    assert opt_res[-1]["cost_eval"] == float(1e4)


def test_normalized_weights_in_pulse(
    dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = get_backend(default_config.backend_config)
    shaper = get_pulse_shaper(simple_qubo_instance, default_config, backend)
    pulse, _ = shaper.generate(dummy_register, simple_qubo_instance)

    norm_weights = pulse.norm_weights
    weights = torch.abs(torch.diag(simple_qubo_instance.coefficients)).tolist()
    max_w = max(weights)
    expected_norm = [1 - (w / max_w) for w in weights]

    assert pytest.approx(norm_weights, rel=1e-6) == expected_norm


def test_pulse_duration_set(
    dummy_register: Register, simple_qubo_instance: QUBOInstance
) -> None:
    default_config = SolverConfig(use_quantum=True)
    backend = get_backend(default_config.backend_config)
    shaper = get_pulse_shaper(simple_qubo_instance, default_config, backend)
    pulse, _ = shaper.generate(dummy_register, simple_qubo_instance)

    assert pulse.duration == 4000


def test_custom_pulse_shaper(simple_qubo_instance: QUBOInstance) -> None:

    class MockAdiabaticPulseShaper(AdiabaticPulseShaper):
        pass

    config = SolverConfig(
        use_quantum=True,
        pulse_shaping=PulseShapingConfig(pulse_shaping_method=MockAdiabaticPulseShaper),
    )
    backend = get_backend(config.backend_config)
    shaper = get_pulse_shaper(simple_qubo_instance, config, backend)
    assert isinstance(shaper, MockAdiabaticPulseShaper)
