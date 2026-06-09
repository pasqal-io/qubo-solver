# Using backends

In `SolverConfig`, we can specify the backend to use when using a quantum approach, that is how we perform quantum runs. Several backends and devices are available via [`Qooqit`](https://github.com/pasqal-io/qoolqit).

## Backend configuration

The backend configuration part in `SolverConfig` is set via two fields.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `backend`     | `LocalEmulator` \| `RemoteEmulator` \| `QPU` | (optional) Which backend to use. |
| `device`      | `Device` | (optional) The quantum device specification. Defaults to `DigitalAnalogDevice`.|


## Local backends

Local backends perform simulations locally with automatic backend selection based on computational tractability. The `LocalEmulator` automatically chooses the appropriate backend:

- **QutipBackendV2**: using the Qutip simulator, for small problems (< 15 qubits) - becomes untractable beyond this limit,
- **SVBackend**: emulator based on state-vector description, for medium problems (15-25 qubits),
- **MPSBackend**: emulator based on state of the art tensor network techniques, for large problems (≥ 26 qubits).

To use the automatic selection, simply instantiate a `SolverConfig` with a `LocalEmulator`:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, LocalEmulator
from qoolqit import DigitalAnalogDevice

# Automatic backend selection based on problem size
config = SolverConfig(
    use_quantum=True,
    backend=LocalEmulator(num_shots=500),
    device=DigitalAnalogDevice(),
)
```

You can also manually specify a particular backend type if needed:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, LocalEmulator
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit import DigitalAnalogDevice

# Manual backend selection
manual_backends = [
    LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
    LocalEmulator(backend_type=SVBackend, num_shots=500),
    LocalEmulator(backend_type=MPSBackend, num_shots=500),
]

config = SolverConfig(
    use_quantum=True,
    backend=manual_backends[0],  # Use QutipBackendV2
    device=DigitalAnalogDevice(),
)
```


## Remote backends

Remote backends submit jobs to a remote server via [pasqal-cloud](https://docs.pasqal.com/cloud/).
The `RemoteEmulator` provides automatic backend selection similar to local emulation, with the same tractability constraints:

- **EmuFreeBackendV2**: remote emulator for small problems (< 15 qubits) - becomes untractable beyond this limit,
- **EmuSVBackend**: remote state-vector emulator for medium problems (15-25 qubits),
- **EmuMPSBackend**: remote tensor network emulator for large problems (≥ 26 qubits).

Note: Fees may apply for remote execution. By default, `RemoteEmulator` uses `EmuFreeBackendV2`.

For this, we require specifying a `RemoteEmulator` or `QPU` and connection details.
Using the code below, replace with your username, project id and password on the Pasqal Cloud.

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, PasqalCloud, RemoteEmulator
from pulser_pasqal.backends import EmuFreeBackendV2, EmuSVBackend, EmuMPSBackend

USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloud(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )

    # Automatic backend selection (recommended)
    config = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(connection=connection, num_shots=500),  # Uses EmuFreeBackendV2 by default
    )

    # Manual backend selection (if needed)
    remote_emulators = [RemoteEmulator(backend_type=btype, connection=connection, num_shots=500)
    for btype in [
        EmuFreeBackendV2,  # For < 15 qubits
        EmuSVBackend,      # For 15-25 qubits
        EmuMPSBackend,     # For ≥ 26 qubits
    ]]
    manual_config = SolverConfig(
        use_quantum=True,
        backend=remote_emulators[0],
    )
```

We can also target a remote QPU as follows:

```python exec="on" source="material-block"
import qoolqit
from qubosolver.config import SolverConfig, PasqalCloud, QPU
from pulser_pasqal.backends import EmuFreeBackendV2, EmuMPSBackend

USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloud(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )
    # specify the QPU device
    device = qoolqit.devices.Device(pulser_device=connection.fetch_available_devices()["FRESNEL"])
    config = SolverConfig(
        use_quantum=True,
        backend = QPU(connection=connection, num_shots=500), device=device,
    )

```

```
