## Custom Pulse Shaper config
If one desires to develop his own pulse shaping method, a subclass of `qubosolver.pipeline.pulse.BasePulseShaper` should be implemented with a mandatory `generate` method.

The `generate` method syntax is `generate(register: Register, instance: QUBOInstance) -> tuple[Pulse, QUBOSolution]`
 with arguments:
- a `Register` instance specifying the qubits we work with.
- a `QUBOInstance` specifying the qubo problem we target.

It returns:
- an instance of `Pulse`
- a `QUBOSolution` specyfing the solution that may be used by a solver.

For concrete examples, we have the [`AdiabaticPulseShaper`](./adiabatic.md) and the [`OptimizedPulseShaper`](./optimized.md) and their current implementations lie in `qubosolver.pipeline.pulse.py`.

Let us show an example of Adiabatic pulse shaper but with a duration divided by 20.

```python exec="on" source="material-block" html="1"
import typing
from qubosolver.pipeline.pulse import BasePulseShaper
from qubosolver.solver import QUBOInstance
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.targets import Register as TargetRegister
from qubosolver.config import (
    PulseShapingConfig,
    SolverConfig,
)
from qubosolver.pipeline.targets import Pulse, Register

from pulser import Pulse as PulserPulse
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform

class LimitedAdiabaticPulseShaper(BasePulseShaper):

    def generate(
        self,
        register: Register,
        instance: QUBOInstance,
    ) -> tuple[Pulse, QUBOSolution]:

        QUBO = instance.coefficients
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list)
        norm_weights_list = [1 - (w / max_node_weight) for w in weights_list]

        T = 4000
        off_diag = QUBO[
            ~torch.eye(QUBO.shape[0], dtype=bool)
        ]  # Selecting off-diagonal terms of the Qubo with a mask
        Omega = min(
            torch.max(off_diag).item(),
            DigitalAnalogDevice.channels["rydberg_global"].max_amp - 1e-9,
        )

        delta_0 = torch.min(torch.diag(QUBO)).item()
        delta_f = -delta_0

        amp_wave = InterpolatedWaveform(T // 20, [1e-9, Omega, 1e-9])
        det_wave = InterpolatedWaveform(T // 20, [delta_0, 0, delta_f])

        pulser_pulse = PulserPulse(amp_wave, det_wave, 0)

        shaped_pulse = Pulse(pulse=pulser_pulse)
        shaped_pulse.norm_weights = norm_weights_list
        shaped_pulse.duration = T

        self.pulse = shaped_pulse
        solution = QUBOSolution(None, None)

        return self.pulse, solution


config = SolverConfig(
    use_quantum=True,
    pulse_shaping=PulseShapingConfig(pulse_shaping_method=LimitedAdiabaticPulseShaper),
)
```
