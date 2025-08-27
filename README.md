# Qubo Solver

Solving combinatorial optimization (CO) problems using quantum computing is one of those promising applications for the near term. The Quadratic Unconstrained Binary Optimization (QUBO) (also known as unconstrained binary quadratic programming) model enables to formulate many CO problems that can be tackled using quantum hardware. QUBO offers a wide range of applications from finance and economics to machine learning.
The Qubo Solver is a Python library designed for solving Quadratic Unconstracined Binary Optimization (QUBO) problems on a neutral atom quantum processor.

The core of the library is focused on the development of several algorithms for solving QUBOs: classical (tabu-search, simulated annealing, ...), quantum (Variational Quantum Algorithms, Quantum Adiabatic Algorithm, ...) or hybrid quantum-classical.

Users setting their first steps into quantum computing will learn how to implement the core algorithm in a few simple steps and run it using the Pasqal Neutral Atom QPU. More experienced users will find this library to provide the right environment to explore new ideas - both in terms of methodologies and data domain - while always interacting with a simple and intuitive QPU interface.

## Development tools

The library uses the following tools:

* [hatch](https://hatch.pypa.io/latest/) for managing virtual environment and dependencies
* [pytest](https://docs.pytest.org/en/7.2.x/contents.html) for building the unit tests suite
* [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/) and [flake8](https://flake8.pycqa.org/en/latest/) for code formatting and linting
* [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
* [pre-commit](https://pre-commit.com/) for applying linting and formatting automatically before committing new code

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


## Document

You can improve the documentation of the package by editing this file for the landing page or adding new
markdown or Jupyter notebooks to the `docs/` folder in the root of the project. In order to modify the
table of contents, edit the `mkdocs.yml` file in the root of the project.

In order to build and serve the documentation locally, you can use `hatch` with the right environment:

```bash
python -m hatch -v run docs:build
python -m hatch -v run docs:serve
```

If you don't want to use `hatch`, just check into your favorite virtual environment and
execute the following commands:

```bash
python -m pip install -r docs/requirements.txt
mkdocs build
mkdocs serve
```

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
