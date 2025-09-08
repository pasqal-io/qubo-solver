# Qubo Solver

Solving combinatorial optimization (CO) problems using quantum computing is one of those promising applications for the near term. The Quadratic Unconstrained Binary Optimization (QUBO) (also known as unconstrained binary quadratic programming) model enables to formulate many CO problems that can be tackled using quantum hardware. QUBO offers a wide range of applications from finance and economics to machine learning.
The Qubo Solver is a Python library designed for solving Quadratic Unconstracined Binary Optimization (QUBO) problems on a neutral atom quantum processor.

The core of the library is focused on the development of several algorithms for solving QUBOs: classical (tabu-search, simulated annealing, ...), quantum (Variational Quantum Algorithms, Quantum Adiabatic Algorithm, ...) or hybrid quantum-classical.

Users setting their first steps into quantum computing will learn how to implement the core algorithm in a few simple steps and run it using the Pasqal Neutral Atom QPU. More experienced users will find this library to provide the right environment to explore new ideas - both in terms of methodologies and data domain - while always interacting with a simple and intuitive QPU interface.

## Development tools

## Installation

### Install as a dependency

Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```
  "qubo-solver"
```

### Using `pip` or `pipx`

To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
$ python -m venv venv

```

2. Enter the venv

```sh
$ . venv/bin/activate
```

3. Install the package

```sh
$ pip install qubo-solver
# or
$ pipx install qubo-solver
```

Alternatively, you can also:

* install with `pip` in development mode by simply running `pip install -e .`. Notice that in this way
  you will install all the dependencies, including extras.
* install it with `conda` by simply using `pip` inside the Conda environment.

### Install on Windows

Note the package is not compatible with Windows systems. We recommend using the Windows Subsystem for Linux (WSL).

### Cplex Installation

The `cplex` dependency may cause installation problems. We recommend using python `3.11` or try another supported version when installing `qubo-solver`.
If you wish to use the licensed version of CPLEX, it is required to set the environment variable `ILOG_LICENSE_FILE` to the location of the license file.


## QuickStart

### With a quantum solver

```python
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

### With a classical solver

```python
from qubosolver import QUBOInstance
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


## Documentation

- [Documentation](https://pasqal-io.github.io/qubo-solver/latest/)
- [Notebooks Tutorials](https://pasqal-io.github.io/qubo-solver/latest/tutorial/01-dataset-generation-and-loading/).
- [Full API documentation](https://pasqal-io.github.io/qubo-solver/latest/api/qubo_instance/).


## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [Github repository](https://github.com/pasqal-io/qubo-solver) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
