# Solving a Quadratic Unconstrained Binary Optimization instance

Solving a QUBO instance is straightforward with `qubo-solver`. We can directly use the `QuboSolver` class by feeding instances of the [`QUBOInstance`](qubo_instance.md) and [`SolverConfig`](config.md) classes. We can also specify whether to use a classical approach or a quantum one.

## Solving with a quantum approach

To use a quantum approach, several choices have to be made regarging the configuration, explained in more details in the [`SolverConfig` section](config.md).
One main decision is about the [backend](backend.md), that is how we choose to perform quantum runs. We can decide to either perform our on emulators (locally, or remotely) or using a real quantum processing unit (QPU). Our QPU, based on the Rydberg Analog Model, is accessible remotely.

### Available backend types and devices

The list of backend types available can be found via the `BackendType` enumeration from [`Qooqit`](https://github.com/pasqal-io/qoolqit), a Python package designed for algorithm development in the Rydberg Analog Model.

```python exec="on" source="material-block" html="1"
from qoolqit._solvers.types import BackendType
all_backends = BackendType.list()
print('Local Backends: ', list(filter(lambda b: 'remote' not in b, all_backends)))
print('Remote Backends: ', list(filter(lambda b: 'remote' in b, all_backends)))
```

The backends can be divided into 3 main categories:

- [Local emulators](https://pasqal-io.github.io/emulators/latest/) (Qutip, Emu_mps, Emu_sv, ...),
- [Remote emulators]((https://docs.pasqal.com/cloud/emu-tn/)), which can be accessed via [`pasqal_cloud`](https://docs.pasqal.com/cloud/),
- [A remote QPU, such as Fresnel](https://docs.pasqal.com/cloud/fresnel-job/).

For emulators, we use device specifications when performing quantum runs via `DeviceType`:

```python exec="on" source="material-block" html="1"
from qoolqit._solvers.types import DeviceType
print([e.value for e in DeviceType])
```

### Running locally with an emulator

We can perform quantum simulations locally via an emulator (here, we choose the `BackendType.QUTIP` emulator).

```python exec="on" source="material-block" html="1"
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.solver import QuboSolver
from qoolqit._solvers.data import BackendConfig
from qoolqit._solvers.types import BackendType

# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# Create a SolverConfig object to use a quantum backend
config = SolverConfig(use_quantum=True, backend_config = BackendConfig(backend=BackendType.QUTIP))

# Instantiate the quantum solver.
solver = QuboSolver(instance, config)

# Solve the QUBO problem.
solution = solver.solve()
```

### Running with a remote connection

We can decide to perform our runs remotely via [`pasqal_cloud`](https://docs.pasqal.com/cloud/).
To do so, we have to provide several information after [setting up an account](https://docs.pasqal.com/cloud/set-up/).

#### On a real QPU

The code above can be modified to solve the QUBO instance using our real QPU remotely as follows:

```python
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.solver import QuboSolver
from qoolqit._solvers.data import BackendConfig
from qoolqit._solvers.types import BackendType, DeviceType

# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# define a remote backend
backendconf = BackendConfig(backend=BackendType.REMOTE_QPU, username='#TO_PROVIDE', password='#TO_PROVIDE', project_id='#TO_PROVIDE')

# Instantiate the quantum solver.
solver = QuboSolver(instance, backend_config=backendconf)

# Solve the QUBO problem.
solution = solver.solve()
```

#### On a remote emulators

Emulators are also available remotely.
Note that the default device set for remote connections is our QPU, but for emulators, you can specify a `DeviceType` as follows:

```python
import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.solver import QuboSolver
from qoolqit._solvers.data import BackendConfig
from qoolqit._solvers.types import BackendType, DeviceType

# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# define a remote backend
backendconf = BackendConfig(backend=BackendType.REMOTE_EMUFREE, device=DeviceType.DIGITAL_ANALOG_DEVICE, username='#TO_PROVIDE', password='#TO_PROVIDE', project_id='#TO_PROVIDE')

# Instantiate the quantum solver.
solver = QuboSolver(instance, backend_config=backendconf)

# Solve the QUBO problem.
solution = solver.solve()
```


## Solving with a classical approach

We show below an example of solving a QUBO using CPLEX.
More information on classical approaches can be found in the `Classical solvers` section of the `Contents` documentation.

```python exec="on" source="material-block" html="1"
import torch
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import ClassicalConfig, SolverConfig
from qubosolver.solver import QuboSolverClassical, QuboSolverQuantum

# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# Create a SolverConfig object with classical solver options.
classical_config = ClassicalConfig(
    classical_solver_type="cplex",
    cplex_maxtime=10.0,
    cplex_log_path="test_solver.log",
)
config = SolverConfig(use_quantum=False, classical=classical_config)

# Instantiate the classical solver via the pipeline's classical solver dispatcher.
classical_solver = QuboSolver(instance, config)

# Solve the QUBO problem.
solution = classical_solver.solve()
```
