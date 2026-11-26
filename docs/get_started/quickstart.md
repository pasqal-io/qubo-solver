## Install from PyPI

Qubo Solver can be installed from [PyPI](https://pypi.org/project/qubo-solver/) with your favorite pyproject-compatible Python manager.
Using `pip`, for example:

```sh
pip install qubo-solver
```

!!! tip "Don't forget to create a virtual environment first!"

##  Your first ⚛️**Quantum**⚛️ solve

!!! info "Runs on an emulator, no QPU required"
    This example runs locally using Pasqal's [emulators](https://docs.pasqal.com/qpu-emulators/emulators/), so you can try it out without access to real quantum hardware.

```python exec="on" session="negative" source="tabbed-left" result="text"
import numpy as np
import qoolqit
from qubosolver import (
    Instance,
    LocalEmulator,
    Solution,
    analysis,
    drive_shaping,
    embedding,
    solving,
)

# Define an Instance from a symmetric matrix
Q = np.asarray(
    [
        [-0.2, 0.0, 1.0],
        [0.0, -1.0, 1.5],
        [1.0, 1.5, -0.1],
    ]
)
instance = Instance(Q)
device = qoolqit.AnalogDeviceWithDMM()

# Map the variables onto atoms, build the laser drive, compile and run on a local emulator
register = embedding.blade.embed_for_device(instance, device)
drive = drive_shaping.local_energy_scale.build_drive(instance, register, device=device)
program = solving.analog_quantum_sampling.compile(register, drive, device)
job = LocalEmulator().run(program)
solution = Solution.from_results(job.results(), instance)

print(analysis.to_dataframe([solution]))
```

The solver returns a [`Solution`][qubosolver.Solution] instance holding the candidate bitstrings and their QUBO costs. As the quantum solver samples the solution space, it also reports each bitstring's count (how many times it was measured) and probability.
