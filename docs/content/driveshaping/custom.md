## Custom Drive Shaper config
If one desires to develop his own drive shaping method, a subclass of `qubosolver.pipeline.drive.BaseDriveShaper` should be implemented with a mandatory `generate` method.

The `generate` method syntax is `generate(register: Register, instance: QUBOInstance) -> tuple[Drive, QUBOSolution]`
 with arguments:

- a `Register` instance specifying the qubits we work with.
- a `QUBOInstance` specifying the qubo problem we target.

It returns:

- an instance of `qoolqit.Drive`
- a `QUBOSolution` specyfing the solution that may be used by a solver.

For concrete examples, we have the [`HeuristicDriveShaper`](./heuristic.md) and the [`OptimizedDriveShaper`](./optimized.md) and their current implementations lie in `qubosolver.pipeline.drive.py`.

Let us show an example of a custom Heuristic drive shaper but with a duration divided by 20.

```python exec="on" source="material-block" html="1"
import typing
from qoolqit import Drive, Register
from qoolqit.waveforms import Interpolated as InterpolatedWaveform
from qubosolver.pipeline.drive import BaseDriveShaper
from qubosolver.solver import QUBOInstance
from qubosolver.data import QUBOSolution
from qubosolver.pipeline.waveforms import weighted_detunings
from qubosolver.config import (
    DriveShapingConfig,
    SolverConfig,
)
from pulser.devices import AnalogDevice

class LimitedAdiabaticDriveShaper(BaseDriveShaper):

    def generate(
        self,
        register: Register,
    ) -> tuple[Drive, QUBOSolution]:

        TIME, ENERGY, _ = self.device.converter.factors

        QUBO = self.instance.coefficients
        weights_list = torch.abs(torch.diag(QUBO)).tolist()
        max_node_weight = max(weights_list)
        norm_weights_list = [(1 - (w / max_node_weight)) / TIME for w in weights_list]

        # enforces AnalogDevice max sequence duration since Digital's one is really specific

        off_diag = QUBO[
            ~torch.eye(QUBO.shape[0], dtype=torch.bool)
        ]  # Selecting off-diagonal terms of the Qubo with a mask

        rydberg_global = self.device._device.channels["rydberg_global"]

        Omega = torch.mean(off_diag).item()
        sign = 1.0 if Omega >= 0 else -1.0
        mag = abs(Omega)
        if min_avg_amp:
            # to make the average values higher then the minimum
            # use the average value of a parabola for
            # the amplitude waveform with Omega
            mag = max(mag, ENERGY * (3.0 * (min_avg_amp + 1e-9) / 2.0))
        if max_amp:
            mag = min(
                mag,
                max_amp - 1e-9,
            )
        Omega = sign * mag

        delta_0 = torch.min(torch.diag(QUBO)).item()
        delta_f = -delta_0

        # enforces AnalogDevice max sequence duration if device has no max
        max_seq_duration = (
            self.device._device.max_sequence_duration or AnalogDevice.max_sequence_duration
        )
        # dividing by 20 here
        max_seq_duration = max_seq_duration // 20

        max_seq_duration /= TIME
        Omega /= TIME
        delta_0 /= TIME
        delta_f /= TIME

        amp_wave = InterpolatedWaveform(max_seq_duration, [1e-9 / TIME, Omega, 1e-9 / TIME])
        det_wave = InterpolatedWaveform(max_seq_duration, [delta_0, 0, delta_f])
        wdetunings = weighted_detunings(
            register,
            max_seq_duration,
            norm_weights_list,
            -delta_f if self.config.drive_shaping.dmm and (delta_f > 0) else None,
        )

        shaped_drive = Drive(amplitude=amp_wave, detuning=det_wave, dmm=wdetunings)
        solution = QUBOSolution(torch.Tensor(), torch.Tensor())

        return shaped_drive, solution


config = SolverConfig(
    use_quantum=True,
    drive_shaping=DriveShapingConfig(drive_shaping_method=LimitedAdiabaticDriveShaper),
)
```
