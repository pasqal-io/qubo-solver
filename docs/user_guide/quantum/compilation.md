# Compilation

Compilation turns the abstract [register](embedding.md) and [drive](drive_shaping.md) into a compiled [`qoolqit.QuantumProgram`][] that respects the target device's hardware constraints, ready to be submitted to a [backend](backend.md).

## Goal of compilation

Embedding and drive shaping produce a `Register` and a `Drive` in dimensionless units, so the same program stays portable across compatible devices. Compilation is the step that converts these dimensionless parameters, times, energies, and distances into their physical equivalents for a target `Device` — using device-dependent scaling based on hardware properties such as the $C_6$ interaction coefficient and the minimum atom spacing — and packages the result into a `QuantumProgram` a backend can execute. Internally, this also generates the low-level [pulser](https://pulser.readthedocs.io/) sequence submitted to the QPU.

`qubosolver` exposes this step as [`solving.analog_quantum_sampling.compile`][qubosolver.solving.quantum.analog_quantum_sampling.compile]:

```python exec="on" source="tabbed-left" result="text"
from qubosolver import Instance, matrix, embedding, drive_shaping, solving
import qoolqit

# Private utility to set seed.
from qubosolver.utils._random import manual_seed
manual_seed(958)

instance = Instance(matrix.tensor([
    [-1, 1, 2, 1],
    [ 1,-3, 3, 0],
    [ 2, 3,-1, 5],
    [ 1, 0, 5,-2],
    ]))
device = qoolqit.AnalogDeviceWithDMM()

register = embedding.blade.embed(instance)
drive = drive_shaping.proportional_diagonal.build_drive(
    instance, register, device=device, dmm=True)

program = solving.analog_quantum_sampling.compile(register, drive, device)
print(program)
```

## Where to go next

The compiled `QuantumProgram` is what gets submitted to a [backend](backend.md) via `backend.run(program)`. For the full rationale behind compilation — dimensionalization, device-dependent scaling, and the available compiler profiles — see [Qoolqit's compilation rationale](https://docs.pasqal.com/qoolqit/fundamentals/compilation/rationale/).
