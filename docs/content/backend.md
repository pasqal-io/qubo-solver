# Using backends

In `SolverConfig`, we can specify the backend to use when using a quantum approach, that is how we perform quantum runs. Several backends and devices are available via [`Qooqit`](https://github.com/pasqal-io/qoolqit).

## Backend configuration

The backend configuration part in `SolverConfig` is set via two fields.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `backend`     | `LocalEmulator` \| `RemoteEmulator` \| `QPU` | (optional) Which backend to use. |
| `device`      | `Device` | (optional) The quantum device specification. Defaults to `AnalogDeviceWithDMM`.|


## Local backends

Local backends perform simulations locally with automatic backend selection based on computational tractability. The `LocalEmulator` automatically chooses the appropriate backend:

- **QutipBackendV2**: using the Qutip simulator, for small problems (< 15 qubits) - becomes intractable beyond this limit,
- **SVBackend**: emulator based on state-vector description, for medium problems (15-25 qubits),
- **MPSBackend**: emulator based on state of the art tensor network techniques, for large problems (≥ 26 qubits).

**Note:** The number of qubits used in the quantum simulation may differ from the input QUBO size due to preprocessing steps or decomposition techniques. For example, the `DecomposeQuboSolver` may break down large QUBO problems into smaller subproblems, affecting the actual number of qubits required for the quantum backend selection.

### Automatic Backend Factory

The automatic backend selection is implemented through `AutoLocalEmulatorBackend`, a factory class that selects the most efficient backend based on the number of qubits.

To use the automatic selection, simply instantiate a `SolverConfig` with a `LocalEmulator`:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, LocalEmulator
from qubosolver.backends import AutoLocalEmulatorBackend
from qoolqit import AnalogDeviceWithDMM

# Automatic backend selection based on problem size (default behavior)
config = SolverConfig(
    use_quantum=True,
    backend=LocalEmulator(num_shots=500),  # Uses AutoLocalEmulatorBackend by default
    device=AnalogDeviceWithDMM(),
)
```

You can also manually specify a particular backend type if needed:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, LocalEmulator
from pulser_simulation import QutipBackendV2
from emu_sv import SVBackend
from emu_mps import MPSBackend
from qoolqit import AnalogDeviceWithDMM

# Manual backend selection
manual_backends = [
    LocalEmulator(backend_type=QutipBackendV2, num_shots=500),
    LocalEmulator(backend_type=SVBackend, num_shots=500),
    LocalEmulator(backend_type=MPSBackend, num_shots=500),
]

config = SolverConfig(
    use_quantum=True,
    backend=manual_backends[0],  # Use QutipBackendV2
    device=AnalogDeviceWithDMM(),
)
```


## Remote backends

Remote backends submit jobs to a remote server via [pasqal-cloud](https://docs.pasqal.com/cloud/).
The `RemoteEmulator` provides backend selection recommendations with the same tractability constraints as local backends:

- **RemoteEmuFreeBackend**: free remote emulator for small problems (< 15 qubits) - becomes intractable beyond this limit,
- **RemoteSVBackend**: paid remote state-vector emulator for medium problems (15-25 qubits),
- **RemoteMPSBackend**: paid remote tensor network emulator for large problems (≥ 26 qubits).

Note: RemoteSVBackend and RemoteMPSBackend incur fees, while RemoteEmuFreeBackend is free. By default, `RemoteEmulator` uses `RemoteEmuFreeBackend`.

### Automatic Remote Backend Factory

For fully automatic remote backend selection, you can use `AutoRemoteEmulatorBackend`, which automatically selects the optimal remote backend based on the number of qubits.

Unlike local emulation, remote emulation typically uses manual backend selection due to cost considerations. Automatic selection is available when needed, but **will incur fees** for problems ≥15 qubits as it automatically switches from the free RemoteEmuFreeBackend to paid backends (RemoteSVBackend or RemoteMPSBackend) based on problem size.

For this, we require specifying a `RemoteEmulator` or `QPU` and connection details.
Using the code below, replace with your username, project id and password on the Pasqal Cloud.

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, PasqalCloudConnection, RemoteEmulator
from qubosolver.backends import AutoRemoteEmulatorBackend
from pasqal_cloud.backends import RemoteEmuFreeBackend, RemoteSVBackend, RemoteMPSBackend

USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloudConnection(
        username=USERNAME,
        password=PASSWORD,
        project_id=PROJECT_ID,
    )

    # Default behavior - uses RemoteEmuFreeBackend with warnings for larger problems
    config_default = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(connection=connection, num_shots=500),
    )

    # Fully automatic backend selection based on problem size
    # WARNING: Will automatically use paid backends for problems ≥15 qubits
    config_auto = SolverConfig(
        use_quantum=True,
        backend=RemoteEmulator(backend_type=AutoRemoteEmulatorBackend, connection=connection, num_shots=500),
    )

    # Manual backend selection (if needed)
    remote_emulators = [RemoteEmulator(backend_type=btype, connection=connection, num_shots=500)
    for btype in [
        RemoteEmuFreeBackend,  # For < 15 qubits
        RemoteSVBackend,      # For 15-25 qubits
        RemoteMPSBackend,     # For ≥ 26 qubits
    ]]
    manual_config = SolverConfig(
        use_quantum=True,
        backend=remote_emulators[0],
    )
```

We can also target a remote QPU as follows:

```python exec="on" source="material-block"
import qoolqit
from qubosolver.config import SolverConfig, PasqalCloudConnection, QPU
from pasqal_cloud.backends import RemoteEmuFreeBackend, RemoteMPSBackend

USERNAME="#TO_PROVIDE"
PROJECT_ID="#TO_PROVIDE"
PASSWORD=None

if PASSWORD is not None:
    connection = PasqalCloudConnection(
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
