# SolverConfig – Solver Configuration Reference

The `SolverConfig` class defines how a QUBO problem should be solved — specifying whether to use a quantum or classical approach, which backend to run on, and additional execution parameters.

This configuration is passed into any solver (e.g., `QuboSolver`) and guides its behavior.
Note that `SolverConfig` uses four other configuration objects: 'BackendConfig', `EmbeddingConfig`, `ClassicalConfig` and `PulseShapingConfig`.
Besides `ClassicalConfig`, the other configurations represents different parts of the solver when using a quantum approach:

---

## Fields for SolverConfig
### Generic parameters
| Field         | Type          | Description |
|---------------|---------------|-------------|
| `config_name` | `str` | The name of the current configuration
| `use_quantum` | `bool` | Whether to solve using a quantum approach (`True`) such as QAA or VQA or a classical approach (`False`). |
| `backend_config`     | `BackendConfig` | Backend part configuration of the solver. |
| `n_calls` | `int` \| `None` | Number of optimization rounds taken to find the best set of parameters for the optimization process inside VQA. The minimum value is 20. Note the optimizer accepts a minimal value of 12. |
| `embedding` | `EmbeddingConfig` | Embedding part configuration of the solver. |
| `pulse_shaping` | `PulseShapingConfig` | Pulse-shaping part configuration of the solver. |
| `classical` | `ClassicalConfig` | Classical part configuration of the solver. |
| `num_shots` | `int` | Number of samples when using a quantum device. Defaults to 500. |

### Backend configuration

The backend configuration part (the `backend_config` field) is set via the `BackendConfig` class. It defines how we will run our quantum programs (via a local emulator, or via remote connection).

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `backend`     | `BackendType` | (optional) Which backend to use (e.g., `'qutip'`, `'emu_mps'`, `'emu_sv'`, `'remote_qpu'`, `'remote_emumps'`). |
| `device`      | `NamedDevice` \| `DeviceType` \| `None` | (optional) If `None`, the backend will pick a reasonable device. If `DeviceType`, choose a device by its capabilities, e.g. `DeviceType.DIGITAL_ANALOG`. If `NamedDevice`, requiest a specific device. Only remote backends make use of `NamedDevice`.|
| `project_id`  | `str` | (optional) Project ID for accessing remote Pasqal services. Only used for remote backends. |
| `username`    | `str` | (optional) Username for Pasqal Cloud authentication. Only used for remote backends. |
| `password`    | `str` | (optional) Password for Pasqal Cloud authentication. Only used for remote backends. |

### Embedding configuration

When solving with a quantum approach, we need to define an embedding method, that is how we define the geometry (register) of atoms based on the QUBO instance and compatibility with a device.
The embedding configuration part (the `embedding` field of `SolverConfig`) can be divided into two groups of parameters.

#### Method parameter
| Field         | Type          | Description |
|---------------|---------------|-------------|
| `embedding_method` | `str` \| `EmbedderType` \| `Type[BaseEmbedder]` | The type of embedding method used to place atoms on the register according to the QUBO problem. (e.g., 'greedy', but we can also create our own custom embedding method). |


#### Greedy embedding parameters

We made available a greedy embedding method (given a fixed lattice or layout, it will defines the register to minimize the incremental mismatch between the logical QUBO matrix Q and the physical device interactions), whose related fields are:

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `layout_greedy_embedder` | `str` \| `LayoutType` \| `None` | Type of layout to run the greedy embedder method on (e.g., 'SquareLatticeLayout', 'TriangularLatticeLayout'). |
| `traps` | `int` \| `None` | The number of traps on the register. |
| `spacing` | `int` \| `None` | The minimum distance between atoms. |
| `density` | `int` \| `None` | The estimated density of the QUBO matrix for the greedy algorithm. |

### Pulse Shaping configuration

Quantum devices can be programmed by specifying a sequence of pulses.
The pulse shaping configuration part (the `pulse_shaping` field of `SolverConfig`) is set via the `PulseShapingConfig` class, and defines how the pulse parameters are constructed (in an adiabatic fashion, via optimization, ...).

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `pulse_shaping_method` | `str` \| `PulseType` \| `Type[BasePulseShaper]` | The type of pulse-shaping method used (e.g., 'adiabatic', 'optimized'). |
| `initial_omega_parameters`   | `list[float]` | The list of initial amplitude $\Omega$ parameters ($3$ floating numbers) to be used in the first round of optimization.|
| `initial_detuning_parameters`   | `list[float]` | The list of global detuning $\delta$ parameters ($3$ floating numbers) to be used in the first round of optimization.|
| `re_execute_opt_pulse` | `bool` | Whether to re-run the optimal pulse sequence. |
| `custom_qubo_cost` | `callable` \| `None` | To apply a different qubo cost evaluation than the default. Must be defined as: `def custom_qubo_cost(bitstring: str, QUBO: torch.Tensor) -> float`. |
| `custom_objective_fn` | `callable` \| `None` | Change the bayesian optimization objective. Instead of using the best cost (`best_cost`) out of the samples, one can change the objective for an average, or any function out of the form `cost_eval = custom_objective_fn(bitstrings, counts, probabilities, costs, best_cost, best_bitstring)` |
| `callback_objective` | `callable` \| `None` | Apply a callback during bayesian optimization. Only accepts one input dictionary created during optimization `d = {"x": x, "cost_eval": cost_eval}` hence should be defined as: `def callback_fn(d: dict) -> None:`. |



### Classical solver configuration

For the classical solver, its configuration can be set via the `ClassicalConfig` class:

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `classical_solver_type`    | `str` | Classical solver type. |
| `cplex_maxtime`    | `float` | CPLEX maximum runtime. |
| `cplex_log_path`    | `str` | CPLEX logging path. |

### Pre-Post processing parameters

We can also apply preprocessing of the QUBO instance (to reduce it to another smaller instance) or postprocessing the solution after solving.

| Field         | Type          | Description |
|---------------|---------------|-------------|
| `do_postprocessing`    | `bool` | Whether we apply post-processing (`True`) or not (`False`). |
| `do_preprocessing`    | `bool` | Whether we apply pre-processing (`True`) or not (`False`). |

---

## Example
The `SolverConfig` is designed in such way that all parameters have a default value which fulfilled the minimum required configuration to execute the necessary steps to solve a QUBO.

All the parameters are `Optional` which allows for running `SolverConfig` without specifying any parameter:
```python exec="on" source="material-block"
from qubosolver.config import SolverConfig
from qoolqit._solvers.types import BackendType, DeviceType
from qubosolver.qubo_types import EmbedderType

config = SolverConfig()
config.print_specs()
```
which returns the following default specifications:
```python
config_name: ''
use_quantum: False
backend: qutip
device: DeviceType.DIGITAL_ANALOG_DEVICE
project_id: ''
username: ''
password: ''
n_calls: 20
embedding: {'embedding_method': <EmbedderType.GREEDY: 'greedy'>, 'layout_greedy_embedder': <LayoutType.SQUARE: <class 'pulser.register.special_layouts.SquareLatticeLayout'>>, 'draw_steps': False, 'traps': 1, 'spacing': 5.0, 'density': None}
pulse_shaping: {'pulse_shaping_method': <PulseType.ADIABATIC: 'adiabatic'>, 'initial_omega_parameters': [5.0, 10.0, 5.0,], 'initial_detuning_parameters': [-10.0, 0.0, 10.0], 're_execute_opt_pulse': False}
classical: {'classical_solver_type': 'cplex', 'cplex_maxtime': 600.0, 'cplex_log_path': 'solver.log'}
do_postprocessing: False
do_preprocessing: False
```
Although the default configuration is straightforward, all parameters can be modified by the user to better suit the specific QUBO instance. Below is an example of a configuration that uses a different embedder with customized parameters on a specific device:
```python exec="on" source="material-block"
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig, EmbeddingConfig, BackendConfig
from qoolqit._solvers.types import DeviceType

coefficients = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)

embedding_config = EmbeddingConfig(embedding_method="greedy", traps=instance.size)
backend_config = BackendConfig(backend="qutip", device=DeviceType.DIGITAL_ANALOG_DEVICE,)

config = SolverConfig(
    config_name="my_config",
    use_quantum=True,
    backend_config = backend_config,
    embedding = embedding_config,
)
```

Equivalently, one can instantiate a `SolverConfig` simply using the keyword arguments of the other configs via the `SolverConfig.from_kwargs` method:

```python exec="on" source="material-block"
from qubosolver import QUBOInstance
from qoolqit._solvers.types import DeviceType
from qubosolver.config import SolverConfig

coefficients = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
instance = QUBOInstance(coefficients=coefficients)

config = SolverConfig.from_kwargs(
    config_name="my_config",
    use_quantum=True,
    backend="qutip",
    device=DeviceType.ANALOG_DEVICE,
    embedding_method="greedy",
    traps=instance.size
)
```
