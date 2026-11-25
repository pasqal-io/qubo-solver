## Custom drive shaping
A drive shaping method is simply a function that builds a [`qoolqit.Drive`][] for a register.
Write your own and pass its output to
[`solving.analog_quantum_sampling.compile`][qubosolver.solving.quantum.analog_quantum_sampling.compile]
like any built-in drive.

For concrete examples, see the [proportional-diagonal](./proportional_diagonal.md) and
[local-energy-scale](./local_energy_scale.md) drives, implemented in
`qubosolver.drive_shaping`.

Let us show an example of a custom simple hard-coded drive.

```python
import numpy as np
import qoolqit
from qoolqit import ConstantWaveform, Drive, RampWaveform

from qubosolver import Instance, LocalEmulator, Solution, embedding, solving


def simple_drive() -> Drive:
    # Defining the drive parameters
    omega = 2.0
    delta_i = -4.0 * omega
    delta_f = -delta_i
    T = 200.0

    # Defining the drive
    wf_amp = ConstantWaveform(T, omega)
    wf_det = RampWaveform(T, delta_i, delta_f)
    return Drive(amplitude=wf_amp, detuning=wf_det)


instance = Instance(np.asarray([[-1.0, 0.5], [0.5, -2.0]]))
device = qoolqit.AnalogDevice()
register = embedding.blade.embed_for_device(instance, device)

program = solving.analog_quantum_sampling.compile(register, simple_drive(), device)
solution = Solution.from_results(LocalEmulator().run(program).results(), instance)
```
