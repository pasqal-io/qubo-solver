# Using backends

In `SolverConfig`, we can specify the backend to use when using a quantum approach. Several backends and devices are available via [`Qooqit`](https://github.com/pasqal-io/qoolqit).

## Backend configuration

The backend configuration part in `SolverConfig` is set via two fields.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `backend`     | `LocalEmulator` \| `RemoteEmulator` \| `QPU` | (optional) Which backend to use. |
| `device`      | `Device` | (optional) The quantum device specification. Defaults to `DigitalAnalogDevice`.|


## Local backends

Local backends perform simulations locally. The main available backends are:

- `qutip` using the Qutip simulator as the default,
- `emu_mps`: emulator based on state of the art tensor network techniques,
- `emu_sv`: emulator based on state-vector description.

To use any, simply instantiate a `SolverConfig` with a `backend` and `device` as follows:

```python exec="on" source="material-backend"
from qubosolver.config import SolverConfig, LocalEmulator
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit import DigitalAnalogDevice

# we can specify the number of shots via the runs attribute
# also, one can configure differently the backends.
# See Qoolqit for more details.
locals_bkds = [
    LocalEmulator(backend_type=btype, runs=500)
    for btype in [
        QutipBackendV2,
        SVBackend,
        MPSBackend,
    ]
]


config = SolverConfig(
    use_quantum=True,
    backend=locals_bkds[0],
    device=DigitalAnalogDevice(),
)
```

Alternatively use the `SolverConfig.from_kwargs` method:

```python exec="on" source="material-backend"

config = SolverConfig.from_kwargs(
    use_quantum=True,
    backend=locals_bkds[0],
    device=DigitalAnalogDevice(),
)
```


## Remote emulators

Remote backends submit jobs to a remote server via [pasqal-cloud](https://docs.pasqal.com/cloud/).
For this, we require specifying a `RemoteEmulator` or `QPU` and connection details.

```python exec="on" source="material-backend"
from qubosolver.config import SolverConfig, PasqalCloud, RemoteEmulator
from pulser_pasqal.backends import EmuFreeBackendV2, EmuMPSBackend

# Replace with your username, project id and password on the Pasqal Cloud.
USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloud(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )
    remote_emulators = [RemoteEmulator(backend_type=btype, connection=connection, runs=500)
    for btype in [
        EmuFreeBackendV2,
        EmuMPSBackend,
    ]]
    config = SolverConfig(
        use_quantum=True,
        backend = remote_emulators[0],
    )
```

We can also target a remote QPU as follows:

```python exec="on" source="material-backend"
from qubosolver.config import SolverConfig, PasqalCloud, QPU
from pulser_pasqal.backends import EmuFreeBackendV2, EmuMPSBackend

# Replace with your username, project id and password on the Pasqal Cloud.
USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloud(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )
    config = SolverConfig(
        use_quantum=True,
        backend = QPU(connection=connection, runs=500),
    )

```
