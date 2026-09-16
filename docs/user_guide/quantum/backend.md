# Backends

A backend runs a compiled [`qoolqit.QuantumProgram`][] — the output of [compilation](compilation.md) — and returns a job you can wait on for results. `qubosolver` doesn't implement its own execution engines; it wraps the backends exposed by [Qoolqit](https://github.com/pasqal-io/qoolqit), picked and configured through `Solver`/`SolverConfig`, or used directly like any other step of the [quantum pipeline](intro.md).

## Goal of backend selection

The same register and drive can be run on several kinds of backends: a local emulator, a remote emulator, or a real QPU. All of them implement the same `run(program)` interface, so switching backends never affects embedding, drive shaping, or compilation — only where and how the program is executed.

Emulators (local or remote) simulate the Rydberg Hamiltonian on classical hardware, so their cost grows quickly with the number of qubits. `qubosolver` ships three emulator engines, each tractable over a different qubit range:

- **Qutip** (`QutipBackendV2` / `RemoteEmuFreeBackend`): exact simulation, tractable for small problems (< 15 qubits) — becomes intractable beyond this limit,
- **State-vector** (`SVBackend` / `RemoteSVBackend`): tractable for medium problems (15-25 qubits),
- **Tensor network** (`MPSBackend` / `RemoteMPSBackend`): state of the art tensor network techniques, tractable for large problems (≥ 26 qubits).

**Note:** The number of qubits used may differ from the input QUBO size due to preprocessing or decomposition — e.g. `DecomposeQuboSolver` breaks large QUBOs into smaller subproblems, changing the qubit count that drives backend selection.

## Local emulator

`LocalEmulator` runs a program on the local machine. Left to pick automatically, it selects the emulator engine best suited to the register's qubit count:

- API reference: [`qubosolver.LocalEmulator`][]
- Tutorial: [Solving a QUBO problem flexibly](../../tutorials/02-qubosolver-in-full.ipynb)

### Code example
```python exec="on" source="tabbed-left" session="backend" result="text"
from qubosolver import Instance, matrix, embedding, drive_shaping, solving, Solution, LocalEmulator, analysis
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
drive = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device, dmm=True)
program = solving.analog_quantum_sampling.compile(register, drive, device)

# Automatic engine selection based on the register's qubit count.
backend = LocalEmulator(num_shots=500)
job = backend.run(program)
solution = Solution.from_results(job.results(), instance)
print(analysis.to_dataframe([solution]))
```

You can also pick a specific engine instead of relying on automatic selection:

```python exec="on" source="tabbed-left" session="backend" result="text"
from qubosolver import LocalEmulator
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend

local_backends = [
    LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
    LocalEmulator(backend_type=SVBackend, num_shots=500),
    LocalEmulator(backend_type=MPSBackend, num_shots=500),
]

job = local_backends[0].run(program)
solution = Solution.from_results(job.results(), instance)
print(analysis.to_dataframe([solution]))
```

## Remote emulator

`RemoteEmulator` submits a program to a remote emulator via [pasqal-cloud](https://docs.pasqal.com/cloud/), rather than running it on the local machine. It offers the same engines as the local emulator, under different tractability trade-offs since remote runs are billed:

- **RemoteEmuFreeBackend**: free, for small problems (< 15 qubits) — becomes intractable beyond this limit; the default,
- **RemoteSVBackend**: paid state-vector emulator, for medium problems (15-25 qubits),
- **RemoteMPSBackend**: paid tensor network emulator, for large problems (≥ 26 qubits).

- API reference: [`qubosolver.RemoteEmulator`][]
- Tutorial: [Solving a QUBO problem flexibly](../../tutorials/02-qubosolver-in-full.ipynb)

Unlike the local emulator, automatic selection here **will incur fees** as soon as the problem reaches 15 qubits, since it switches away from the free engine — so `RemoteEmulator` defaults to the free engine rather than automatic selection. Opt into automatic selection explicitly via `AutoRemoteEmulatorBackend` if you want it.

### Code example

Replace `USERNAME`, `PROJECT_ID` and `PASSWORD` with your [Pasqal Cloud](https://docs.pasqal.com/cloud/set-up/) credentials to run this example against the real cloud. Without a password, it falls back to a `LocalConnection`, which emulates the same interface locally with QuTiP — no credentials needed, handy for trying the snippet out or for tests.

```python exec="on" source="tabbed-left" session="backend" result="text"
from qubosolver import RemoteEmulator, AutoRemoteEmulatorBackend, Solution, analysis
from pasqal_cloud import PasqalCloudConnection
from pasqal_cloud.backends import RemoteEmuFreeBackend, RemoteSVBackend, RemoteMPSBackend

USERNAME = "#TO_PROVIDE"
PROJECT_ID = "#TO_PROVIDE"
PASSWORD = None

if PASSWORD is not None:
    connection = PasqalCloudConnection(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )
else:
    # Local fallback so this snippet runs without credentials.
    from qubosolver.utils._local_connection import LocalConnection
    connection = LocalConnection()

# Default: the free engine, regardless of problem size.
backend = RemoteEmulator(connection=connection, num_shots=500)

# Automatic engine selection based on problem size.
# WARNING: will use paid engines for problems >= 15 qubits.
auto_backend = RemoteEmulator(
    backend_type=AutoRemoteEmulatorBackend, connection=connection, num_shots=500)

# Manual engine selection (only supported by a real cloud connection).
remote_backends = [
    RemoteEmulator(backend_type=btype, connection=connection, num_shots=500)
    for btype in [RemoteEmuFreeBackend, RemoteSVBackend, RemoteMPSBackend]
]

job = backend.run(program)
solution = Solution.from_results(job.results(), instance)
print(analysis.to_dataframe([solution]))
```

## QPU

`qoolqit.execution.QPU` submits a program to a real Pasqal QPU (e.g. Fresnel), accessed remotely through the same `pasqal-cloud` connection as the remote emulator. Unlike emulators, the device specification must match a QPU actually available on your account, fetched from the connection rather than instantiated directly.

- API reference: [`qoolqit.execution.QPU`][]
- Tutorial: [Solving a QUBO problem flexibly](../../tutorials/02-qubosolver-in-full.ipynb)

### Code example

Submitting to a real QPU requires a `PasqalCloudConnection` — there is no local equivalent of the hardware itself. Without a password, the snippet below falls back to emulating the same instance locally with `LocalEmulator` instead, so it stays runnable without credentials.

```python exec="on" source="tabbed-left" result="text"
from qubosolver import Instance, LocalEmulator, matrix, embedding, drive_shaping, solving, Solution, analysis
from qoolqit.execution import QPU
from pasqal_cloud import PasqalCloudConnection
import qoolqit

USERNAME = "#TO_PROVIDE"
PROJECT_ID = "#TO_PROVIDE"
PASSWORD = None

instance = Instance(matrix.tensor([
    [-1, 1, 2, 1],
    [ 1,-3, 3, 0],
    [ 2, 3,-1, 5],
    [ 1, 0, 5,-2],
    ]))

if PASSWORD is not None:
    connection = PasqalCloudConnection(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )
    print(f"Available devices: {connection.fetch_available_devices()}")
    device = qoolqit.Device.from_connection(connection, "FRESNEL")
    backend = QPU(connection=connection, num_shots=500)
else:
    # Local fallback so this snippet runs without credentials.
    device = qoolqit.AnalogDeviceWithDMM()
    backend = LocalEmulator(num_shots=500)

register = embedding.blade.embed(instance)
drive = drive_shaping.proportional_diagonal.build_drive(instance, register, device=device, dmm=True)
program = solving.analog_quantum_sampling.compile(register, drive, device)

job = backend.run(program)
solution = Solution.from_results(job.results(), instance)
print(analysis.to_dataframe([solution]))
```

## Asynchronous remote runs

Remote runs, whether on a QPU or a remote emulator, are submitted asynchronously: `backend.run(program)` returns as soon as the job is queued, without waiting for results. This lets you save the job's identifiers and the instance, disconnect, and retrieve the results later — from the same session or a different one — rather than blocking until the run completes. See the [qubosolver-in-full tutorial](../../tutorials/02-qubosolver-in-full.ipynb) for the full save/retrieve example.

## The `Solver` shortcut

Rather than building a program and calling `backend.run` directly, you can select and configure the backend and device through `QuantumSolvingConfig`, nested in `SolverConfig`; `Solver` then runs it as part of the full quantum pipeline. Leaving `backend`/`device` unset falls back to a `LocalEmulator` on `AnalogDeviceWithDMM` — see [`SolverConfig`][qubosolver.SolverConfig] for the full set of defaults.

```python exec="on" source="tabbed-left" session="backend" result="text"
from qubosolver import SolverConfig, QuantumSolvingConfig, LocalEmulator
from pulser_simulation import QutipBackendV2
from dataclasses import asdict
import pprint
import qoolqit

quantum_config = QuantumSolvingConfig(
    backend=LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
    device=qoolqit.AnalogDeviceWithDMM(),
)
solver_config = SolverConfig(
    solving=quantum_config,
)
print(pprint.pformat(asdict(solver_config.solving)))
```
