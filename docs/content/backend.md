# Using backends

In `SolverConfig`, we can specify the backend to use when using a quantum approach. Several backends are available via [`Qooqit`](https://github.com/pasqal-io/qoolqit).

## Backend configuration

The backend configuration part is set via the `BackendConfig` class.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `backend`     | `BackendType` | (optional) Which backend to use (e.g., `'qutip'`, `'emu_mps'`, `'emu_sv'`, `'remote_qpu'`, `'remote_emumps'`). |
| `username`    | `str` | (optional) Username for Pasqal Cloud authentication. Only used for remote backends. |
| `password`    | `str` | (optional) Password for Pasqal Cloud authentication. Only used for remote backends. |
| `project_id`  | `str` | (optional) Project ID for accessing remote Pasqal services. Only used for remote backends. |
| `device`      | `NamedDevice` \| `DeviceType` \| `None` | (optional) If `None`, the backend will pick a reasonable device. If `DeviceType`, choose a device by its capabilities, e.g. `DeviceType.DIGITAL_ANALOG`. If `NamedDevice`, requiest a specific device. Only remote backends make use of `NamedDevice`.|
| `dt`  | `float` | (optional) For a backend that supports customizing the duration of steps, the timestep size can be provided. As of this writing, this parameter is used only by the EmuMPS backends. |
| `wait`  | `bool` | (optional) For a remote backend where we submit a batch of jobs, block execution on this statement until all the submitted jobs are terminated. Defaults to False. |


## Local backends

Local backends perform simulations locally. The available backends are:

- `qutip` using the Qutip simulator,
- `emu_mps`: emulator based on state of the art tensor network techniques,
- `emu_sv`: emulator based on state-vector description.

To use any, simply instantiate a `SolverConfig` with a `BackendConfig` as follows:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, BackendConfig
from qoolqit._solvers.types import DeviceType

backend_config = BackendConfig(backend="qutip", device=DeviceType.DIGITAL_ANALOG_DEVICE,)

config = SolverConfig(
    use_quantum=True,
    backend_config = backend_config,
)
```

Alternatively use the `SolverConfig.from_kwargs` method with the `BackendConfig` parameters:

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig
from qoolqit._solvers.types import DeviceType

config = SolverConfig.from_kwargs(
    use_quantum=True,
    backend="qutip",
    device=DeviceType.DIGITAL_ANALOG_DEVICE
)
```


## Remote backends

Remote backends submit jobs to a remote server via [pasqal-cloud](https://docs.pasqal.com/cloud/).
For this, we require specifying in `BackendConfig` the `project_id`, `username` and `password`.
We can target a remote QPU, or emulator among the choices:

- `remote_qpu`,
- `remote_emumps`,
- `remote_emutn`,
- `remote_emufree`: remote emulator based on `Qutip`.

```python exec="on" source="material-block"
from qubosolver.config import SolverConfig, BackendConfig
from qoolqit._solvers.types import DeviceType

backend_config = BackendConfig(backend="remote_emufree", device=DeviceType.DIGITAL_ANALOG_DEVICE, project_id='pid', username='admin', password='pwd')

config = SolverConfig(
    use_quantum=True,
    backend_config = backend_config,
)
```
